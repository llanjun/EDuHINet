import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, f1_score, accuracy_score, classification_report,
    confusion_matrix, recall_score, fbeta_score, brier_score_loss,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed}")


class TimeSeriesDataset(Dataset):
    def __init__(self, temporal_data, static_data, labels):
        self.temporal_data = torch.FloatTensor(temporal_data)
        self.static_data = torch.FloatTensor(static_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.temporal_data[idx], self.static_data[idx], self.labels[idx]


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class BilinearInteraction(nn.Module):
    def __init__(self, dim1, dim2, output_dim):
        super(BilinearInteraction, self).__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, output_dim)

    def forward(self, x1, x2):
        return self.bilinear(x1, x2)


class DeepMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(DeepMLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln3 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        identity = self.residual_proj(x)

        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.ln3(out)
        out = F.relu(out)

        out = out + identity
        return out


class HeterogeneousModel(nn.Module):
    def __init__(self, temporal_feature_dim, static_feature_dim,
                 lstm_hidden_dim=128, fc_hidden_dim=64,
                 mlp_hidden_dim=128, num_classes=4, dropout_rate=0.3):
        super(HeterogeneousModel, self).__init__()

        self.lstm = nn.LSTM(temporal_feature_dim, lstm_hidden_dim,
                            num_layers=2, batch_first=True,
                            dropout=dropout_rate)

        self.attention = AttentionLayer(lstm_hidden_dim)

        self.static_mlp = DeepMLPBlock(static_feature_dim, 128, fc_hidden_dim, dropout_rate)

        self.bilinear = BilinearInteraction(lstm_hidden_dim, fc_hidden_dim, mlp_hidden_dim)

        fusion_input_dim = lstm_hidden_dim + fc_hidden_dim + mlp_hidden_dim
        self.fusion_mlp1 = DeepMLPBlock(fusion_input_dim, mlp_hidden_dim * 2, mlp_hidden_dim, dropout_rate)

        fusion2_input_dim = lstm_hidden_dim + fc_hidden_dim + mlp_hidden_dim * 2
        self.fusion_mlp2 = DeepMLPBlock(fusion2_input_dim, mlp_hidden_dim * 2, mlp_hidden_dim, dropout_rate)

        self.output_fc = nn.Linear(mlp_hidden_dim, num_classes)

    def forward(self, temporal_data, static_data):
        lstm_out, _ = self.lstm(temporal_data)

        C, attention_weights = self.attention(lstm_out)

        S = self.static_mlp(static_data)

        bilinear_out = self.bilinear(C, S)
        Z = torch.cat([C, S, bilinear_out], dim=1)

        u1 = self.fusion_mlp1(Z)

        fusion2 = torch.cat([C, S, bilinear_out, u1], dim=1)

        u2 = self.fusion_mlp2(fusion2)

        output = self.output_fc(u2)

        return output


def load_and_preprocess_data(excel_path):
    print("=" * 50)
    print("Loading data...")
    df = pd.read_excel(excel_path)
    print(f"Dataset size: {df.shape}")
    print(f"Label distribution:\n{df['final_result'].value_counts().sort_index()}")

    temporal_cols = []
    static_cols = []

    for col in df.columns:
        if col.startswith('timestep'):
            temporal_cols.append(col)
        elif col != 'final_result':
            static_cols.append(col)

    print(f"\nNumber of non-temporal features: {len(static_cols)}")
    print(f"Total number of temporal features: {len(temporal_cols)}")

    timesteps = {}
    for col in temporal_cols:
        timestep_num = int(col.split('_')[0].replace('timestep', ''))
        if timestep_num not in timesteps:
            timesteps[timestep_num] = []
        timesteps[timestep_num].append(col)

    num_timesteps = len(timesteps)
    max_features = max(len(features) for features in timesteps.values())

    print(f"\nNumber of timesteps: {num_timesteps}")
    for t in sorted(timesteps.keys()):
        print(f"  Timestep {t}: {len(timesteps[t])} features")
    print(f"Max features: {max_features}")

    num_samples = len(df)
    temporal_data = np.zeros((num_samples, num_timesteps, max_features))

    for t_idx, t in enumerate(sorted(timesteps.keys())):
        features = timesteps[t]
        feature_values = df[features].values
        temporal_data[:, t_idx, :len(features)] = feature_values

    print(f"\nTemporal data shape: {temporal_data.shape}")

    static_data = df[static_cols].values if static_cols else np.zeros((num_samples, 1))
    print(f"Non-temporal data shape: {static_data.shape}")

    labels = df['final_result'].values

    temporal_data_scaled = temporal_data
    static_data_scaled = static_data

    print("=" * 50)

    return temporal_data_scaled, static_data_scaled, labels, max_features, static_data.shape[1]


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for temporal, static, labels in train_loader:
        temporal, static, labels = temporal.to(device), static.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(temporal, static)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, criterion, device, show_detail=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for temporal, static, labels in val_loader:
            temporal, static, labels = temporal.to(device), static.to(device), labels.to(device)

            outputs = model(temporal, static)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f0_5_macro = fbeta_score(all_labels, all_preds, beta=0.5, average='macro', zero_division=0)
    f2_macro = fbeta_score(all_labels, all_preds, beta=2.0, average='macro', zero_division=0)

    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f0_5_micro = fbeta_score(all_labels, all_preds, beta=0.5, average='micro', zero_division=0)
    f2_micro = fbeta_score(all_labels, all_preds, beta=2.0, average='micro', zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f0_5_weighted = fbeta_score(all_labels, all_preds, beta=0.5, average='weighted', zero_division=0)
    f2_weighted = fbeta_score(all_labels, all_preds, beta=2.0, average='weighted', zero_division=0)

    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    f0_5_per_class = fbeta_score(all_labels, all_preds, beta=0.5, average=None, zero_division=0)
    f2_per_class = fbeta_score(all_labels, all_preds, beta=2.0, average=None, zero_division=0)

    accuracy_per_class = recall_per_class

    brier_scores = []
    for i in range(all_probs.shape[1]):
        y_true_binary = (all_labels == i).astype(float)
        y_prob = all_probs[:, i]
        brier_scores.append(brier_score_loss(y_true_binary, y_prob))
    brier_score_avg = np.mean(brier_scores)

    class_metrics = {
        'precision': precision_per_class,
        'recall': recall_per_class,
        'accuracy': accuracy_per_class,
        'f1': f1_per_class,
        'f0_5': f0_5_per_class,
        'f2': f2_per_class,
        'brier': np.array(brier_scores)
    }

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f0_5_macro': f0_5_macro,
        'f2_macro': f2_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'f0_5_micro': f0_5_micro,
        'f2_micro': f2_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'f0_5_weighted': f0_5_weighted,
        'f2_weighted': f2_weighted,
        'brier_score': brier_score_avg,
        'class_metrics': class_metrics
    }

    if show_detail:
        print("\n" + "="*50)
        print("Per-class detailed report:")
        print("="*50)
        print(classification_report(all_labels, all_preds,
                                   target_names=['Distinction', 'Pass', 'Fail', 'Withdrawn'],
                                   digits=4))

        print("\nConfusion matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

    return total_loss / len(val_loader), accuracy, precision_macro, f1_macro, all_preds, all_labels, class_metrics, metrics, all_probs


def plot_roc_curve_with_ci(y_true, y_probs, class_names=None, save_path=None, n_bootstrap=1000, ci=0.95):
    n_classes = y_probs.shape[1]
    if class_names is None:
        class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(10, 8))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    aucs_bootstrap = []

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        n_samples = len(y_true)
        auc_bootstrap = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true_binary[indices]
            y_score_boot = y_score[indices]

            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
                auc_boot = auc(fpr_boot, tpr_boot)
                auc_bootstrap.append(auc_boot)
            except:
                continue

        if auc_bootstrap:
            aucs_bootstrap.append(auc_bootstrap)
            alpha = 1 - ci
            lower = np.percentile(auc_bootstrap, alpha/2 * 100)
            upper = np.percentile(auc_bootstrap, (1 - alpha/2) * 100)
            print(f"{class_names[i]}: AUC = {roc_auc:.4f} ({ci*100:.0f}% CI: [{lower:.4f}, {upper:.4f}])")

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    if aucs_bootstrap:
        macro_auc_bootstrap = [np.mean([aucs_bootstrap[c][b] for c in range(n_classes) if b < len(aucs_bootstrap[c])])
                               for b in range(min(len(a) for a in aucs_bootstrap))]
        if macro_auc_bootstrap:
            alpha = 1 - ci
            macro_lower = np.percentile(macro_auc_bootstrap, alpha/2 * 100)
            macro_upper = np.percentile(macro_auc_bootstrap, (1 - alpha/2) * 100)
            print(f"Macro-average: AUC = {mean_auc:.4f} ({ci*100:.0f}% CI: [{macro_lower:.4f}, {macro_upper:.4f}])")

    std_tpr = np.std(tprs, axis=0)
    ax.plot(mean_fpr, mean_tpr, color='navy', lw=2, linestyle='--',
            label=f'Macro-average (AUC = {mean_auc:.3f})')

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='navy', alpha=0.2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves with 95% Confidence Interval', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")

    plt.close()

    return aucs, mean_auc


def plot_roc_curves_per_class(y_true, y_probs, class_names=None, save_dir='results', n_bootstrap=1000, ci=0.95):
    from sklearn.metrics import roc_curve, auc

    n_classes = y_probs.shape[1]
    if class_names is None:
        class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    auc_results = {}

    print("\n" + "="*60)
    print("Per-class ROC curve analysis (One-vs-Rest, with 95% CI)")
    print("="*60)

    for i in range(n_classes):
        print(f"\nProcessing: {class_names[i]}...")

        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        n_samples = len(y_true)
        tprs_boot = []
        aucs_boot = []

        np.random.seed(42)
        for b in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true_binary[indices]
            y_score_boot = y_score[indices]

            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
                auc_boot = auc(fpr_boot, tpr_boot)
                aucs_boot.append(auc_boot)

                interp_tpr = np.interp(np.linspace(0, 1, 100), fpr_boot, tpr_boot)
                interp_tpr[0] = 0.0
                interp_tpr[-1] = 1.0
                tprs_boot.append(interp_tpr)
            except:
                continue

        alpha = 1 - ci
        auc_lower = np.percentile(aucs_boot, alpha/2 * 100)
        auc_upper = np.percentile(aucs_boot, (1 - alpha/2) * 100)

        auc_results[class_names[i]] = {
            'auc': roc_auc,
            'ci_lower': auc_lower,
            'ci_upper': auc_upper
        }

        print(f"  {class_names[i]}: AUC = {roc_auc:.4f} ({ci*100:.0f}% CI: [{auc_lower:.4f}, {auc_upper:.4f}])")

        mean_tpr_boot = np.mean(tprs_boot, axis=0)
        std_tpr_boot = np.std(tprs_boot, axis=0)

        tpr_upper = np.minimum(mean_tpr_boot + std_tpr_boot, 1)
        tpr_lower = np.maximum(mean_tpr_boot - std_tpr_boot, 0)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.fill_between(np.linspace(0, 1, 100), tpr_lower, tpr_upper,
                        color=colors[i], alpha=0.3, label=f'{ci*100:.0f}% CI')

        ax.plot(fpr, tpr, color=colors[i], lw=2.5,
                label=f'{class_names[i]} (AUC = {roc_auc:.4f}, 95% CI: [{auc_lower:.4f}, {auc_upper:.4f}])')

        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5000)')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {class_names[i]}\n(One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f'{save_dir}/ROC_curve_{class_names[i].replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Chart saved to: {save_path}")
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path_all = f'{save_dir}/ROC_curves_all_classes_comparison.png'
    plt.savefig(save_path_all, dpi=300, bbox_inches='tight')
    print(f"\nAll classes comparison plot saved to: {save_path_all}")
    plt.close()

    print("\n" + "="*60)
    print("Per-class ROC curve analysis completed!")
    print("="*60)

    return auc_results


def plot_confusion_matrices(all_labels, all_preds, class_names=None, save_dir='results', n_classes=4):
    if class_names is None:
        class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']

    cm = confusion_matrix(all_labels, all_preds)

    for i in range(n_classes):
        y_true_binary = (all_labels == i).astype(int)
        y_pred_binary = (all_preds == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        cm_binary = np.array([[tp, fn], [fp, tn]])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Positive', 'Negative'],
                    yticklabels=['Positive', 'Negative'],
                    annot_kws={'size': 14},
                    cbar=False)

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{class_names[i]}', fontsize=14, fontweight='bold')

        ax.text(0.5, -0.15, f'TN={tn}, FP={fp}    FN={fn}, TP={tp}',
                ha='center', transform=ax.transAxes, fontsize=11)

        plt.tight_layout()
        save_path_single = f'{save_dir}/confusion_matrix_{class_names[i].replace(" ", "_")}.png'
        plt.savefig(save_path_single, dpi=300, bbox_inches='tight')
        print(f"{class_names[i]} confusion matrix saved to: {save_path_single}")
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 14})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (EDuHINet)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path_overall = f'{save_dir}/confusion_matrix_overall.png'
    plt.savefig(save_path_overall, dpi=300, bbox_inches='tight')
    print(f"Overall confusion matrix saved to: {save_path_overall}")
    plt.close()

    return cm


def compute_sensitivity_analysis(model, device, temporal_data, static_data, feature_info, n_samples=100, perturbation_ratio=0.1):
    import copy

    print("\n" + "="*50)
    print("Starting sensitivity analysis...")
    print("="*50)

    model.eval()

    n, T, F = temporal_data.shape
    S = static_data.shape[1]

    temporal_flat = temporal_data.reshape(n, -1)
    X_combined = np.hstack([temporal_flat, static_data])
    num_features = X_combined.shape[1]

    np.random.seed(42)
    sample_idx = np.random.choice(n, min(n_samples, n), replace=False)
    X_samples = X_combined[sample_idx].copy()

    feature_stds = np.std(X_combined, axis=0)
    feature_stds[feature_stds == 0] = 1

    original_probs = []
    with torch.no_grad():
        for i in range(len(X_samples)):
            x = X_samples[i]
            x_tensor = torch.FloatTensor(x).to(device)
            temporal_part = x_tensor[: T * F].reshape(1, T, F)
            static_part = x_tensor[T * F :].reshape(1, S)

            out = model(temporal_part, static_part)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            original_probs.append(probs)

    original_probs = np.array(original_probs)

    sensitivity_scores = np.zeros(num_features)

    class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']
    sensitivity_per_class = {name: np.zeros(num_features) for name in class_names}

    for feat_idx in range(num_features):
        if feat_idx % 10 == 0:
            print(f"  Processing feature {feat_idx + 1}/{num_features}...")

        perturbation = feature_stds[feat_idx] * perturbation_ratio

        X_perturbed_pos = X_samples.copy()
        X_perturbed_pos[:, feat_idx] += perturbation

        X_perturbed_neg = X_samples.copy()
        X_perturbed_neg[:, feat_idx] -= perturbation

        perturbed_probs_pos = []
        perturbed_probs_neg = []

        with torch.no_grad():
            for i in range(len(X_perturbed_pos)):
                x = X_perturbed_pos[i]
                x_tensor = torch.FloatTensor(x).to(device)
                temporal_part = x_tensor[: T * F].reshape(1, T, F)
                static_part = x_tensor[T * F :].reshape(1, S)

                out = model(temporal_part, static_part)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                perturbed_probs_pos.append(probs)

            for i in range(len(X_perturbed_neg)):
                x = X_perturbed_neg[i]
                x_tensor = torch.FloatTensor(x).to(device)
                temporal_part = x_tensor[: T * F].reshape(1, T, F)
                static_part = x_tensor[T * F :].reshape(1, S)

                out = model(temporal_part, static_part)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                perturbed_probs_neg.append(probs)

        perturbed_probs_pos = np.array(perturbed_probs_pos)
        perturbed_probs_neg = np.array(perturbed_probs_neg)

        change_pos = np.abs(perturbed_probs_pos - original_probs)
        change_neg = np.abs(perturbed_probs_neg - original_probs)
        avg_change = (change_pos + change_neg) / 2

        sensitivity_scores[feat_idx] = np.mean(avg_change)

        for class_idx, class_name in enumerate(class_names):
            sensitivity_per_class[class_name][feat_idx] = np.mean(avg_change[:, class_idx])

    sensitivity_scores_normalized = sensitivity_scores / (np.max(sensitivity_scores) + 1e-10)

    for class_name in class_names:
        max_val = np.max(sensitivity_per_class[class_name])
        if max_val > 0:
            sensitivity_per_class[class_name] = sensitivity_per_class[class_name] / max_val

    all_names, temporal_names, static_names = build_feature_names(feature_info)
    feature_names = all_names[: num_features]

    print(f"Sensitivity analysis completed!")
    print(f"  Highest sensitivity feature: {feature_names[np.argmax(sensitivity_scores)]}")
    print(f"  Highest sensitivity value: {np.max(sensitivity_scores):.6f}")

    return sensitivity_scores, sensitivity_scores_normalized, sensitivity_per_class, feature_names


def build_feature_names(feature_info):
    timesteps = feature_info["timesteps"]
    static_cols = feature_info["static_cols"]

    temporal_feature_names = []
    for t in sorted(timesteps.keys()):
        for col in timesteps[t]:
            if "_" in col:
                feat = col.split("_", 1)[1]
            else:
                feat = col
            temporal_feature_names.append(f"T{t}_{feat}")

    static_feature_names = [f"S_{col}" for col in static_cols]
    all_names = temporal_feature_names + static_feature_names
    return all_names, temporal_feature_names, static_feature_names


def plot_sensitivity_analysis(sensitivity_scores, sensitivity_normalized, sensitivity_per_class,
                               feature_names, temporal_names, static_names, save_dir='results'):
    print("\n" + "="*50)
    print("Plotting sensitivity analysis...")
    print("="*50)

    df_sensitivity = pd.DataFrame({
        'Feature': feature_names,
        'Sensitivity': sensitivity_scores,
        'Sensitivity_Normalized': sensitivity_normalized
    })
    df_sensitivity_sorted = df_sensitivity.sort_values('Sensitivity', ascending=False)

    top_k = min(20, len(df_sensitivity_sorted))
    top_features = df_sensitivity_sorted.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.4)))
    ax.barh(range(len(top_features)), top_features['Sensitivity'], color='#e74c3c')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=10)
    ax.set_xlabel('Sensitivity Score (Mean |DeltaProbability|)', fontsize=11)
    ax.set_title('Top 20 Most Sensitive Features (Sensitivity Analysis)', fontsize=12)
    plt.tight_layout()
    save_path1 = f'{save_dir}/sensitivity_top20.png'
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"Top 20 sensitivity plot saved to: {save_path1}")
    plt.close()

    df_temporal = df_sensitivity[df_sensitivity['Feature'].str.startswith('T')].copy()
    df_temporal_sorted = df_temporal.sort_values('Sensitivity', ascending=False)

    top_temporal = min(10, len(df_temporal_sorted))
    if top_temporal > 0:
        top_temporal_df = df_temporal_sorted.head(top_temporal).iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, max(3, top_temporal * 0.4)))
        ax.barh(range(len(top_temporal_df)), top_temporal_df['Sensitivity'], color='#3498db')
        ax.set_yticks(range(len(top_temporal_df)))
        ax.set_yticklabels(top_temporal_df['Feature'], fontsize=10)
        ax.set_xlabel('Sensitivity Score', fontsize=11)
        ax.set_title('Top 10 Most Sensitive Temporal Features', fontsize=12)
        plt.tight_layout()
        save_path2 = f'{save_dir}/sensitivity_temporal_top10.png'
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"Top 10 temporal sensitivity plot saved to: {save_path2}")
        plt.close()

    df_static = df_sensitivity[df_sensitivity['Feature'].str.startswith('S')].copy()
    df_static_sorted = df_static.sort_values('Sensitivity', ascending=False)

    top_static = min(5, len(df_static_sorted))
    if top_static > 0:
        top_static_df = df_static_sorted.head(top_static).iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, max(3, top_static * 0.4)))
        ax.barh(range(len(top_static_df)), top_static_df['Sensitivity'], color='#2ecc71')
        ax.set_yticks(range(len(top_static_df)))
        ax.set_yticklabels(top_static_df['Feature'], fontsize=10)
        ax.set_xlabel('Sensitivity Score', fontsize=11)
        ax.set_title('Top 5 Most Sensitive Static Features', fontsize=12)
        plt.tight_layout()
        save_path3 = f'{save_dir}/sensitivity_static_top5.png'
        plt.savefig(save_path3, dpi=300, bbox_inches='tight')
        print(f"Top 5 static sensitivity plot saved to: {save_path3}")
        plt.close()

    class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']
    top_features_idx = df_sensitivity_sorted.head(15).index.tolist()

    sensitivity_heatmap = np.array([
        [sensitivity_per_class[name][idx] for idx in top_features_idx]
        for name in class_names
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(sensitivity_heatmap, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(top_features_idx)))
    ax.set_xticklabels([feature_names[i] for i in top_features_idx], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=10)

    plt.colorbar(im, ax=ax, label='Normalized Sensitivity')
    ax.set_title('Sensitivity Heatmap: Top 15 Features x 4 Classes', fontsize=12)
    plt.tight_layout()
    save_path4 = f'{save_dir}/sensitivity_heatmap.png'
    plt.savefig(save_path4, dpi=300, bbox_inches='tight')
    print(f"Sensitivity heatmap saved to: {save_path4}")
    plt.close()

    df_sensitivity_full = df_sensitivity_sorted.reset_index(drop=True)
    excel_path = f'{save_dir}/sensitivity_analysis_results.xlsx'
    df_sensitivity_full.to_excel(excel_path, index=False)
    print(f"Full sensitivity results saved to: {excel_path}")

    return df_sensitivity_sorted


def compute_stability_analysis(model, device, temporal_data, static_data, n_runs=10):
    print("\n" + "="*50)
    print("Starting stability analysis...")
    print("="*50)

    model.eval()

    n, T, F = temporal_data.shape
    S = static_data.shape[1]

    np.random.seed(42)
    n_test = min(50, n)
    test_idx = np.random.choice(n, n_test, replace=False)

    temporal_test = temporal_data[test_idx]
    static_test = static_data[test_idx]

    all_predictions = []

    for run in range(n_runs):
        predictions = []

        with torch.no_grad():
            for i in range(n_test):
                temporal = torch.FloatTensor(temporal_test[i:i+1]).to(device)
                static = torch.FloatTensor(static_test[i:i+1]).to(device)

                out = model(temporal, static)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                predictions.append(probs)

        all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)

    prediction_variance = np.var(all_predictions, axis=0)
    mean_variance_per_class = np.mean(prediction_variance, axis=0)

    prediction_std = np.std(all_predictions, axis=0)
    mean_std_per_class = np.mean(prediction_std, axis=0)

    overall_variance = np.mean(prediction_variance)
    overall_std = np.mean(prediction_std)

    print(f"Stability analysis results ({n_runs} predictions):")
    class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']
    for i, name in enumerate(class_names):
        print(f"  {name}: Mean Std = {mean_std_per_class[i]:.6f}, Mean Variance = {mean_variance_per_class[i]:.6f}")
    print(f"  Overall: Mean Std = {overall_std:.6f}, Mean Variance = {overall_variance:.6f}")

    return {
        'mean_variance_per_class': mean_variance_per_class,
        'mean_std_per_class': mean_std_per_class,
        'overall_variance': overall_variance,
        'overall_std': overall_std,
        'prediction_variance': prediction_variance
    }


def train_model_with_cv(temporal_data, static_data, labels,
                        temporal_feature_dim, static_feature_dim,
                        n_splits=10, epochs=50, batch_size=32, lr=0.001, seed=42):
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []
    all_fold_probs = []
    all_fold_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(temporal_data, labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")

        X_temp_train, X_temp_val = temporal_data[train_idx], temporal_data[val_idx]
        X_stat_train, X_stat_val = static_data[train_idx], static_data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_dataset = TimeSeriesDataset(X_temp_train, X_stat_train, y_train)
        val_dataset = TimeSeriesDataset(X_temp_val, X_stat_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = HeterogeneousModel(
            temporal_feature_dim=temporal_feature_dim,
            static_feature_dim=static_feature_dim,
            lstm_hidden_dim=128,
            fc_hidden_dim=64,
            mlp_hidden_dim=128,
            num_classes=4,
            dropout_rate=0.3
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0
        best_metrics = {}
        best_preds = None
        best_labels = None
        best_probs = None

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_precision, val_f1, val_preds, val_labels, class_metrics, val_metrics, val_probs = evaluate(
                model, val_loader, criterion, device, show_detail=False)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = val_metrics
                best_preds = val_preds
                best_labels = val_labels
                best_probs = val_probs
                torch.save(model.state_dict(), f'results/best_model_v2_fold{fold+1}.pth')

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        print(f"\nFold {fold + 1} Best results:")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  Macro Precision: {best_metrics['precision_macro']:.4f}")
        print(f"  Macro Recall: {best_metrics['recall_macro']:.4f}")
        print(f"  Macro F1: {best_metrics['f1_macro']:.4f}")
        print(f"  Macro F0.5: {best_metrics['f0_5_macro']:.4f}")
        print(f"  Macro F2: {best_metrics['f2_macro']:.4f}")
        print(f"  Micro Precision: {best_metrics['precision_micro']:.4f}")
        print(f"  Micro Recall: {best_metrics['recall_micro']:.4f}")
        print(f"  Micro F1: {best_metrics['f1_micro']:.4f}")
        print(f"  Weighted F1: {best_metrics['f1_weighted']:.4f}")
        print(f"  Brier Score: {best_metrics['brier_score']:.4f}")

        if best_preds is not None:
            print("\n" + "="*50)
            print(f"Fold {fold + 1} Per-class detailed report:")
            print("="*50)
            print(classification_report(best_labels, best_preds,
                                       target_names=['Distinction', 'Pass', 'Fail', 'Withdrawn'],
                                       digits=4))

            print("\nConfusion matrix:")
            cm = confusion_matrix(best_labels, best_preds)
            print(cm)

        fold_results.append(best_metrics)
        if best_probs is not None:
            all_fold_probs.append(best_probs)
            all_fold_labels.append(best_labels)

    print(f"\n{'='*50}")
    print("10-Fold Cross-Validation Average Results:")
    print(f"{'='*50}")
    avg_acc = np.mean([r['accuracy'] for r in fold_results])

    avg_precision_macro = np.mean([r['precision_macro'] for r in fold_results])
    avg_recall_macro = np.mean([r['recall_macro'] for r in fold_results])
    avg_f1_macro = np.mean([r['f1_macro'] for r in fold_results])
    avg_f0_5_macro = np.mean([r['f0_5_macro'] for r in fold_results])
    avg_f2_macro = np.mean([r['f2_macro'] for r in fold_results])

    avg_precision_micro = np.mean([r['precision_micro'] for r in fold_results])
    avg_recall_micro = np.mean([r['recall_micro'] for r in fold_results])
    avg_f1_micro = np.mean([r['f1_micro'] for r in fold_results])
    avg_f0_5_micro = np.mean([r['f0_5_micro'] for r in fold_results])
    avg_f2_micro = np.mean([r['f2_micro'] for r in fold_results])

    avg_precision_weighted = np.mean([r['precision_weighted'] for r in fold_results])
    avg_recall_weighted = np.mean([r['recall_weighted'] for r in fold_results])
    avg_f1_weighted = np.mean([r['f1_weighted'] for r in fold_results])
    avg_f0_5_weighted = np.mean([r['f0_5_weighted'] for r in fold_results])
    avg_f2_weighted = np.mean([r['f2_weighted'] for r in fold_results])

    avg_brier = np.mean([r['brier_score'] for r in fold_results])

    print(f"\n[Overall Metrics]")
    print(f"Accuracy: {avg_acc:.4f} +/- {np.std([r['accuracy'] for r in fold_results]):.4f}")
    print(f"\n[Macro Average]")
    print(f"  Precision: {avg_precision_macro:.4f} +/- {np.std([r['precision_macro'] for r in fold_results]):.4f}")
    print(f"  Recall:    {avg_recall_macro:.4f} +/- {np.std([r['recall_macro'] for r in fold_results]):.4f}")
    print(f"  F1 Score: {avg_f1_macro:.4f} +/- {np.std([r['f1_macro'] for r in fold_results]):.4f}")
    print(f"  F0.5:     {avg_f0_5_macro:.4f} +/- {np.std([r['f0_5_macro'] for r in fold_results]):.4f}")
    print(f"  F2:       {avg_f2_macro:.4f} +/- {np.std([r['f2_macro'] for r in fold_results]):.4f}")
    print(f"\n[Micro Average]")
    print(f"  Precision: {avg_precision_micro:.4f} +/- {np.std([r['precision_micro'] for r in fold_results]):.4f}")
    print(f"  Recall:    {avg_recall_micro:.4f} +/- {np.std([r['recall_micro'] for r in fold_results]):.4f}")
    print(f"  F1 Score: {avg_f1_micro:.4f} +/- {np.std([r['f1_micro'] for r in fold_results]):.4f}")
    print(f"  F0.5:     {avg_f0_5_micro:.4f} +/- {np.std([r['f0_5_micro'] for r in fold_results]):.4f}")
    print(f"  F2:       {avg_f2_micro:.4f} +/- {np.std([r['f2_micro'] for r in fold_results]):.4f}")
    print(f"\n[Weighted Average]")
    print(f"  Precision: {avg_precision_weighted:.4f} +/- {np.std([r['precision_weighted'] for r in fold_results]):.4f}")
    print(f"  Recall:    {avg_recall_weighted:.4f} +/- {np.std([r['recall_weighted'] for r in fold_results]):.4f}")
    print(f"  F1 Score: {avg_f1_weighted:.4f} +/- {np.std([r['f1_weighted'] for r in fold_results]):.4f}")
    print(f"  F0.5:     {avg_f0_5_weighted:.4f} +/- {np.std([r['f0_5_weighted'] for r in fold_results]):.4f}")
    print(f"  F2:       {avg_f2_weighted:.4f} +/- {np.std([r['f2_weighted'] for r in fold_results]):.4f}")
    print(f"\nBrier Score: {avg_brier:.4f} +/- {np.std([r['brier_score'] for r in fold_results]):.4f}")

    print(f"\n[Per-Class Average Metrics]")
    class_precision_all = []
    class_recall_all = []
    class_accuracy_all = []
    class_f1_all = []
    class_f0_5_all = []
    class_f2_all = []
    class_brier_all = []

    for r in fold_results:
        class_precision_all.append(r['class_metrics']['precision'])
        class_recall_all.append(r['class_metrics']['recall'])
        class_accuracy_all.append(r['class_metrics']['accuracy'])
        class_f1_all.append(r['class_metrics']['f1'])
        class_f0_5_all.append(r['class_metrics']['f0_5'])
        class_f2_all.append(r['class_metrics']['f2'])
        class_brier_all.append(r['class_metrics']['brier'])

    class_precision_mean = np.mean(class_precision_all, axis=0)
    class_precision_std = np.std(class_precision_all, axis=0)
    class_recall_mean = np.mean(class_recall_all, axis=0)
    class_recall_std = np.std(class_recall_all, axis=0)
    class_accuracy_mean = np.mean(class_accuracy_all, axis=0)
    class_accuracy_std = np.std(class_accuracy_all, axis=0)
    class_f1_mean = np.mean(class_f1_all, axis=0)
    class_f1_std = np.std(class_f1_all, axis=0)
    class_f0_5_mean = np.mean(class_f0_5_all, axis=0)
    class_f0_5_std = np.std(class_f0_5_all, axis=0)
    class_f2_mean = np.mean(class_f2_all, axis=0)
    class_f2_std = np.std(class_f2_all, axis=0)
    class_brier_mean = np.mean(class_brier_all, axis=0)
    class_brier_std = np.std(class_brier_all, axis=0)

    class_names_display = ['Distinction', 'Pass', 'Fail', 'Withdrawn']

    for i in range(4):
        print(f"\n  {class_names_display[i]}:")
        print(f"    Precision: {class_precision_mean[i]:.4f} +/- {class_precision_std[i]:.4f}")
        print(f"    Recall:    {class_recall_mean[i]:.4f} +/- {class_recall_std[i]:.4f}")
        print(f"    Accuracy:  {class_accuracy_mean[i]:.4f} +/- {class_accuracy_std[i]:.4f}")
        print(f"    F1 Score:  {class_f1_mean[i]:.4f} +/- {class_f1_std[i]:.4f}")
        print(f"    F0.5:      {class_f0_5_mean[i]:.4f} +/- {class_f0_5_std[i]:.4f}")
        print(f"    F2:        {class_f2_mean[i]:.4f} +/- {class_f2_std[i]:.4f}")
        print(f"    Brier:     {class_brier_mean[i]:.4f} +/- {class_brier_std[i]:.4f}")

    excel_data = []

    excel_data.append({'Metric': 'Accuracy', 'Value': f"{avg_acc:.4f}+/-{np.std([r['accuracy'] for r in fold_results]):.4f}"})

    excel_data.append({'Metric': 'Macro Precision', 'Value': f"{avg_precision_macro:.4f}+/-{np.std([r['precision_macro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Macro Recall', 'Value': f"{avg_recall_macro:.4f}+/-{np.std([r['recall_macro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Macro F1 Score', 'Value': f"{avg_f1_macro:.4f}+/-{np.std([r['f1_macro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Macro F0.5 Score', 'Value': f"{avg_f0_5_macro:.4f}+/-{np.std([r['f0_5_macro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Macro F2 Score', 'Value': f"{avg_f2_macro:.4f}+/-{np.std([r['f2_macro'] for r in fold_results]):.4f}"})

    excel_data.append({'Metric': 'Micro Precision', 'Value': f"{avg_precision_micro:.4f}+/-{np.std([r['precision_micro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Micro Recall', 'Value': f"{avg_recall_micro:.4f}+/-{np.std([r['recall_micro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Micro F1 Score', 'Value': f"{avg_f1_micro:.4f}+/-{np.std([r['f1_micro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Micro F0.5 Score', 'Value': f"{avg_f0_5_micro:.4f}+/-{np.std([r['f0_5_micro'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Micro F2 Score', 'Value': f"{avg_f2_micro:.4f}+/-{np.std([r['f2_micro'] for r in fold_results]):.4f}"})

    excel_data.append({'Metric': 'Weighted Precision', 'Value': f"{avg_precision_weighted:.4f}+/-{np.std([r['precision_weighted'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Weighted Recall', 'Value': f"{avg_recall_weighted:.4f}+/-{np.std([r['recall_weighted'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Weighted F1 Score', 'Value': f"{avg_f1_weighted:.4f}+/-{np.std([r['f1_weighted'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Weighted F0.5 Score', 'Value': f"{avg_f0_5_weighted:.4f}+/-{np.std([r['f0_5_weighted'] for r in fold_results]):.4f}"})
    excel_data.append({'Metric': 'Weighted F2 Score', 'Value': f"{avg_f2_weighted:.4f}+/-{np.std([r['f2_weighted'] for r in fold_results]):.4f}"})

    excel_data.append({'Metric': 'Brier Score', 'Value': f"{avg_brier:.4f}+/-{np.std([r['brier_score'] for r in fold_results]):.4f}"})

    class_names_excel = ['Distinction', 'Pass', 'Fail', 'Withdrawn']

    for i in range(4):
        excel_data.append({'Metric': f'{class_names_excel[i]} Precision', 'Value': f"{class_precision_mean[i]:.4f}+/-{class_precision_std[i]:.4f}"})
        excel_data.append({'Metric': f'{class_names_excel[i]} Recall', 'Value': f"{class_recall_mean[i]:.4f}+/-{class_recall_std[i]:.4f}"})
        excel_data.append({'Metric': f'{class_names_excel[i]} Accuracy', 'Value': f"{class_accuracy_mean[i]:.4f}+/-{class_accuracy_std[i]:.4f}"})
        excel_data.append({'Metric': f'{class_names_excel[i]} F1 Score', 'Value': f"{class_f1_mean[i]:.4f}+/-{class_f1_std[i]:.4f}"})
        excel_data.append({'Metric': f'{class_names_excel[i]} F0.5 Score', 'Value': f"{class_f0_5_mean[i]:.4f}+/-{class_f0_5_std[i]:.4f}"})
        excel_data.append({'Metric': f'{class_names_excel[i]} F2 Score', 'Value': f"{class_f2_mean[i]:.4f}+/-{class_f2_std[i]:.4f}"})
        excel_data.append({'Metric': f'{class_names_excel[i]} Brier Score', 'Value': f"{class_brier_mean[i]:.4f}+/-{class_brier_std[i]:.4f}"})

    df_results = pd.DataFrame(excel_data)
    excel_filename = 'results/results_HeterogeneousModel_V2_DeepMLP.xlsx'
    df_results.to_excel(excel_filename, index=False)
    print(f"\nResults saved to: {excel_filename}")

    if all_fold_probs and all_fold_labels:
        print("\n" + "="*50)
        print("Plotting ROC curves (with 95% CI)...")
        print("="*50)

        all_probs_combined = np.vstack(all_fold_probs)
        all_labels_combined = np.concatenate(all_fold_labels)

        class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']
        roc_save_path = 'results/ROC_curve_with_CI.png'

        aucs, mean_auc = plot_roc_curve_with_ci(
            y_true=all_labels_combined,
            y_probs=all_probs_combined,
            class_names=class_names,
            save_path=roc_save_path,
            n_bootstrap=1000,
            ci=0.95
        )

        print("\n" + "="*50)
        print("Plotting per-class ROC curves (One-vs-Rest, with 95% CI)...")
        print("="*50)

        auc_results_per_class = plot_roc_curves_per_class(
            y_true=all_labels_combined,
            y_probs=all_probs_combined,
            class_names=class_names,
            save_dir='results',
            n_bootstrap=1000,
            ci=0.95
        )

        print("\n" + "="*50)
        print("Plotting confusion matrices...")
        print("="*50)

        cm = plot_confusion_matrices(
            all_labels=all_labels_combined,
            all_preds=np.argmax(all_probs_combined, axis=1),
            class_names=class_names,
            save_dir='results',
            n_classes=4
        )

    return fold_results


if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)

    excel_path = "final_data_processed.xlsx"
    temporal_data, static_data, labels, temporal_feature_dim, static_feature_dim = load_and_preprocess_data(excel_path)

    results = train_model_with_cv(
        temporal_data=temporal_data,
        static_data=static_data,
        labels=labels,
        temporal_feature_dim=temporal_feature_dim,
        static_feature_dim=static_feature_dim,
        n_splits=10,
        epochs=50,
        batch_size=32,
        lr=0.001,
        seed=SEED
    )

    print("\n" + "="*50)
    print("Loading best model for sensitivity analysis...")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HeterogeneousModel(
        temporal_feature_dim=temporal_feature_dim,
        static_feature_dim=static_feature_dim,
        lstm_hidden_dim=128,
        fc_hidden_dim=64,
        mlp_hidden_dim=128,
        num_classes=4,
        dropout_rate=0.3
    ).to(device)

    model.load_state_dict(torch.load('results/best_model_v2_fold1.pth'))

    feature_info = {}
    df = pd.read_excel(excel_path)
    temporal_cols = [col for col in df.columns if col.startswith('timestep')]
    static_cols = [col for col in df.columns if col != 'final_result']

    timesteps = {}
    for col in temporal_cols:
        timestep_num = int(col.split('_')[0].replace('timestep', ''))
        timesteps.setdefault(timestep_num, []).append(col)

    feature_info = {
        'temporal_cols': temporal_cols,
        'static_cols': static_cols,
        'timesteps': timesteps,
        'num_timesteps': len(timesteps),
        'max_features': max(len(v) for v in timesteps.values())
    }

    sensitivity_scores, sensitivity_normalized, sensitivity_per_class, feature_names = compute_sensitivity_analysis(
        model=model,
        device=device,
        temporal_data=temporal_data,
        static_data=static_data,
        feature_info=feature_info,
        n_samples=100,
        perturbation_ratio=0.1
    )

    all_names, temporal_names, static_names = build_feature_names(feature_info)
    df_sensitivity = plot_sensitivity_analysis(
        sensitivity_scores=sensitivity_scores,
        sensitivity_normalized=sensitivity_normalized,
        sensitivity_per_class=sensitivity_per_class,
        feature_names=feature_names,
        temporal_names=temporal_names,
        static_names=static_names,
        save_dir='results'
    )

    stability_results = compute_stability_analysis(
        model=model,
        device=device,
        temporal_data=temporal_data,
        static_data=static_data,
        n_runs=10
    )

    stability_df = pd.DataFrame({
        'Class': ['Distinction', 'Pass', 'Fail', 'Withdrawn', 'Overall'],
        'Mean_Variance': list(stability_results['mean_variance_per_class']) + [stability_results['overall_variance']],
        'Mean_Std': list(stability_results['mean_std_per_class']) + [stability_results['overall_std']]
    })
    stability_df.to_excel('results/stability_analysis_results.xlsx', index=False)
    print(f"Stability analysis results saved to: results/stability_analysis_results.xlsx")

    print("\nTraining completed! Model weights saved.")
    print("Sensitivity analysis and stability analysis completed!")
