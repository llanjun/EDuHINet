import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, f1_score, recall_score, fbeta_score, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def load_data(excel_path):
    print("=" * 50)
    print("Loading data...")
    df = pd.read_excel(excel_path)
    print(f"Dataset size: {df.shape}")
    print(f"Label distribution:\n{df['final_result'].value_counts().sort_index()}")

    X = df.drop('final_result', axis=1).values
    y = df['final_result'].values

    print(f"\nNumber of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print("=" * 50)

    return X, y


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    accuracy = accuracy_score(y_test, y_pred)

    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f0_5_macro = fbeta_score(y_test, y_pred, beta=0.5, average='macro', zero_division=0)
    f2_macro = fbeta_score(y_test, y_pred, beta=2.0, average='macro', zero_division=0)

    precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f0_5_micro = fbeta_score(y_test, y_pred, beta=0.5, average='micro', zero_division=0)
    f2_micro = fbeta_score(y_test, y_pred, beta=2.0, average='micro', zero_division=0)

    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f0_5_weighted = fbeta_score(y_test, y_pred, beta=0.5, average='weighted', zero_division=0)
    f2_weighted = fbeta_score(y_test, y_pred, beta=2.0, average='weighted', zero_division=0)

    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    accuracy_per_class = recall_per_class
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    f0_5_per_class = fbeta_score(y_test, y_pred, beta=0.5, average=None, zero_division=0)
    f2_per_class = fbeta_score(y_test, y_pred, beta=2.0, average=None, zero_division=0)

    brier_scores = []
    if y_prob is not None:
        for i in range(y_prob.shape[1]):
            y_true_binary = (y_test == i).astype(float)
            y_prob_class = y_prob[:, i]
            brier_scores.append(brier_score_loss(y_true_binary, y_prob_class))
    brier_score_avg = np.mean(brier_scores) if brier_scores else 0.0

    class_metrics = {
        'precision': precision_per_class,
        'recall': recall_per_class,
        'accuracy': accuracy_per_class,
        'f1': f1_per_class,
        'f0_5': f0_5_per_class,
        'f2': f2_per_class,
        'brier': np.array(brier_scores) if brier_scores else np.zeros(4)
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

    return accuracy, precision_macro, f1_macro, y_pred, class_metrics, metrics


def cross_validate_models(X, y, n_splits=10, random_seed=42):
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_seed,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'CatBoost': CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            loss_function='MultiClass',
            random_seed=random_seed,
            verbose=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_seed,
            verbose=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=random_seed,
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_seed
        )
    }

    X_scaled = X

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    all_results = {name: [] for name in models.keys()}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            if model_name == 'CatBoost':
                model.fit(X_train, y_train, verbose=False)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                f0_5_macro = fbeta_score(y_test, y_pred, beta=0.5, average='macro', zero_division=0)
                f2_macro = fbeta_score(y_test, y_pred, beta=2.0, average='macro', zero_division=0)

                precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
                recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
                f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
                f0_5_micro = fbeta_score(y_test, y_pred, beta=0.5, average='micro', zero_division=0)
                f2_micro = fbeta_score(y_test, y_pred, beta=2.0, average='micro', zero_division=0)

                precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                f0_5_weighted = fbeta_score(y_test, y_pred, beta=0.5, average='weighted', zero_division=0)
                f2_weighted = fbeta_score(y_test, y_pred, beta=2.0, average='weighted', zero_division=0)

                precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
                recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
                accuracy_per_class = recall_per_class
                f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
                f0_5_per_class = fbeta_score(y_test, y_pred, beta=0.5, average=None, zero_division=0)
                f2_per_class = fbeta_score(y_test, y_pred, beta=2.0, average=None, zero_division=0)

                brier_scores = []
                for i in range(y_prob.shape[1]):
                    y_true_binary = (y_test == i).astype(float)
                    brier_scores.append(brier_score_loss(y_true_binary, y_prob[:, i]))
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
            else:
                accuracy, precision_macro, f1_macro, y_pred, class_metrics, metrics = train_and_evaluate_model(
                    model, model_name, X_train, X_test, y_train, y_test
                )

            all_results[model_name].append({
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'f1_macro': f1_macro,
                'y_pred': y_pred,
                'y_test': y_test,
                'class_metrics': class_metrics,
                'metrics': metrics
            })

            print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision(Macro): {precision_macro:.4f}, F1(Macro): {f1_macro:.4f}")

    return all_results


def plot_confusion_matrix_for_model(all_labels, all_preds, model_name,
                                    class_names=None, save_dir='results',
                                    n_classes=4):
    if class_names is None:
        class_names = ['Distinction', 'Pass', 'Fail', 'Withdrawn']

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 14}, linewidths=0.5,
                linecolor='white', cbar=False)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_overall = f'{save_dir}/confusion_matrix_{model_name}_overall.png'
    plt.savefig(save_overall, dpi=300, bbox_inches='tight')
    print(f"  [{model_name}] Overall confusion matrix saved: {save_overall}")
    plt.close()

    for i in range(n_classes):
        y_true_binary = (all_labels == i).astype(int)
        y_pred_binary = (all_preds == i).astype(int)
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        if cm_binary.shape == (2, 2):
            tn, fp, fn, tp = cm_binary.ravel()
        else:
            tn = fp = fn = tp = 0

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    annot_kws={'size': 14}, linewidths=0.5,
                    linecolor='white', cbar=False)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{model_name} - {class_names[i]} (OvR)', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.18, f'TN={tn}, FP={fp}    FN={fn}, TP={tp}',
                ha='center', transform=ax.transAxes, fontsize=10)
        plt.tight_layout()
        save_binary = f'{save_dir}/confusion_matrix_{model_name}_Class{i}_{class_names[i].replace(" ", "_")}.png'
        plt.savefig(save_binary, dpi=300, bbox_inches='tight')
        print(f"  [{model_name}] {class_names[i]} binary confusion matrix saved: {save_binary}")
        plt.close()

    return cm


def print_summary(all_results):
    print(f"\n{'='*50}")
    print("10-Fold Cross-Validation Summary Results")
    print(f"{'='*50}\n")

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print("-" * 50)

        accuracies = [r['accuracy'] for r in results]
        precisions_macro = [r['metrics']['precision_macro'] for r in results]
        f1s_macro = [r['metrics']['f1_macro'] for r in results]
        f0_5s_macro = [r['metrics']['f0_5_macro'] for r in results]
        f2s_macro = [r['metrics']['f2_macro'] for r in results]

        precisions_micro = [r['metrics']['precision_micro'] for r in results]
        f1s_micro = [r['metrics']['f1_micro'] for r in results]

        precisions_weighted = [r['metrics']['precision_weighted'] for r in results]
        f1s_weighted = [r['metrics']['f1_weighted'] for r in results]

        brier_scores = [r['metrics']['brier_score'] for r in results]

        print(f"\n[Overall Metrics]")
        print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"\n[Macro Average]")
        print(f"  Precision: {np.mean(precisions_macro):.4f} +/- {np.std(precisions_macro):.4f}")
        print(f"  Recall:    {np.mean([r['metrics']['recall_macro'] for r in results]):.4f} +/- {np.std([r['metrics']['recall_macro'] for r in results]):.4f}")
        print(f"  F1 Score: {np.mean(f1s_macro):.4f} +/- {np.std(f1s_macro):.4f}")
        print(f"  F0.5:     {np.mean(f0_5s_macro):.4f} +/- {np.std(f0_5s_macro):.4f}")
        print(f"  F2:       {np.mean(f2s_macro):.4f} +/- {np.std(f2s_macro):.4f}")
        print(f"\n[Micro Average]")
        print(f"  Precision: {np.mean(precisions_micro):.4f} +/- {np.std(precisions_micro):.4f}")
        print(f"  Recall:    {np.mean([r['metrics']['recall_micro'] for r in results]):.4f} +/- {np.std([r['metrics']['recall_micro'] for r in results]):.4f}")
        print(f"  F1 Score: {np.mean(f1s_micro):.4f} +/- {np.std(f1s_micro):.4f}")
        print(f"  F0.5:     {np.mean([r['metrics']['f0_5_micro'] for r in results]):.4f} +/- {np.std([r['metrics']['f0_5_micro'] for r in results]):.4f}")
        print(f"  F2:       {np.mean([r['metrics']['f2_micro'] for r in results]):.4f} +/- {np.std([r['metrics']['f2_micro'] for r in results]):.4f}")
        print(f"\n[Weighted Average]")
        print(f"  Precision: {np.mean(precisions_weighted):.4f} +/- {np.std(precisions_weighted):.4f}")
        print(f"  Recall:    {np.mean([r['metrics']['recall_weighted'] for r in results]):.4f} +/- {np.std([r['metrics']['recall_weighted'] for r in results]):.4f}")
        print(f"  F1 Score: {np.mean(f1s_weighted):.4f} +/- {np.std(f1s_weighted):.4f}")
        print(f"  F0.5:     {np.mean([r['metrics']['f0_5_weighted'] for r in results]):.4f} +/- {np.std([r['metrics']['f0_5_weighted'] for r in results]):.4f}")
        print(f"  F2:       {np.mean([r['metrics']['f2_weighted'] for r in results]):.4f} +/- {np.std([r['metrics']['f2_weighted'] for r in results]):.4f}")
        print(f"\nBrier Score: {np.mean(brier_scores):.4f} +/- {np.std(brier_scores):.4f}")

        print(f"\n[Per-Class Average Metrics]")
        class_precision_all = []
        class_recall_all = []
        class_accuracy_all = []
        class_f1_all = []
        class_f0_5_all = []
        class_f2_all = []
        class_brier_all = []

        for r in results:
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

        for i in range(4):
            print(f"\n  Class {i}:")
            print(f"    Precision: {class_precision_mean[i]:.4f} +/- {class_precision_std[i]:.4f}")
            print(f"    Recall:    {class_recall_mean[i]:.4f} +/- {class_recall_std[i]:.4f}")
            print(f"    Accuracy:  {class_accuracy_mean[i]:.4f} +/- {class_accuracy_std[i]:.4f}")
            print(f"    F1 Score:  {class_f1_mean[i]:.4f} +/- {class_f1_std[i]:.4f}")
            print(f"    F0.5:      {class_f0_5_mean[i]:.4f} +/- {class_f0_5_std[i]:.4f}")
            print(f"    F2:        {class_f2_mean[i]:.4f} +/- {class_f2_std[i]:.4f}")
            print(f"    Brier:     {class_brier_mean[i]:.4f} +/- {class_brier_std[i]:.4f}")

        excel_data = []

        excel_data.append({'Metric': 'Accuracy', 'Value': f"{np.mean(accuracies):.4f}+/-{np.std(accuracies):.4f}"})

        excel_data.append({'Metric': 'Macro Precision', 'Value': f"{np.mean(precisions_macro):.4f}+/-{np.std(precisions_macro):.4f}"})
        excel_data.append({'Metric': 'Macro Recall', 'Value': f"{np.mean([r['metrics']['recall_macro'] for r in results]):.4f}+/-{np.std([r['metrics']['recall_macro'] for r in results]):.4f}"})
        excel_data.append({'Metric': 'Macro F1 Score', 'Value': f"{np.mean(f1s_macro):.4f}+/-{np.std(f1s_macro):.4f}"})
        excel_data.append({'Metric': 'Macro F0.5 Score', 'Value': f"{np.mean(f0_5s_macro):.4f}+/-{np.std(f0_5s_macro):.4f}"})
        excel_data.append({'Metric': 'Macro F2 Score', 'Value': f"{np.mean(f2s_macro):.4f}+/-{np.std(f2s_macro):.4f}"})

        excel_data.append({'Metric': 'Micro Precision', 'Value': f"{np.mean(precisions_micro):.4f}+/-{np.std(precisions_micro):.4f}"})
        excel_data.append({'Metric': 'Micro Recall', 'Value': f"{np.mean([r['metrics']['recall_micro'] for r in results]):.4f}+/-{np.std([r['metrics']['recall_micro'] for r in results]):.4f}"})
        excel_data.append({'Metric': 'Micro F1 Score', 'Value': f"{np.mean(f1s_micro):.4f}+/-{np.std(f1s_micro):.4f}"})
        excel_data.append({'Metric': 'Micro F0.5 Score', 'Value': f"{np.mean([r['metrics']['f0_5_micro'] for r in results]):.4f}+/-{np.std([r['metrics']['f0_5_micro'] for r in results]):.4f}"})
        excel_data.append({'Metric': 'Micro F2 Score', 'Value': f"{np.mean([r['metrics']['f2_micro'] for r in results]):.4f}+/-{np.std([r['metrics']['f2_micro'] for r in results]):.4f}"})

        excel_data.append({'Metric': 'Weighted Precision', 'Value': f"{np.mean(precisions_weighted):.4f}+/-{np.std(precisions_weighted):.4f}"})
        excel_data.append({'Metric': 'Weighted Recall', 'Value': f"{np.mean([r['metrics']['recall_weighted'] for r in results]):.4f}+/-{np.std([r['metrics']['recall_weighted'] for r in results]):.4f}"})
        excel_data.append({'Metric': 'Weighted F1 Score', 'Value': f"{np.mean(f1s_weighted):.4f}+/-{np.std(f1s_weighted):.4f}"})
        excel_data.append({'Metric': 'Weighted F0.5 Score', 'Value': f"{np.mean([r['metrics']['f0_5_weighted'] for r in results]):.4f}+/-{np.std([r['metrics']['f0_5_weighted'] for r in results]):.4f}"})
        excel_data.append({'Metric': 'Weighted F2 Score', 'Value': f"{np.mean([r['metrics']['f2_weighted'] for r in results]):.4f}+/-{np.std([r['metrics']['f2_weighted'] for r in results]):.4f}"})

        excel_data.append({'Metric': 'Brier Score', 'Value': f"{np.mean(brier_scores):.4f}+/-{np.std(brier_scores):.4f}"})

        for i in range(4):
            excel_data.append({'Metric': f'Class {i} Precision', 'Value': f"{class_precision_mean[i]:.4f}+/-{class_precision_std[i]:.4f}"})
            excel_data.append({'Metric': f'Class {i} Recall', 'Value': f"{class_recall_mean[i]:.4f}+/-{class_recall_std[i]:.4f}"})
            excel_data.append({'Metric': f'Class {i} Accuracy', 'Value': f"{class_accuracy_mean[i]:.4f}+/-{class_accuracy_std[i]:.4f}"})
            excel_data.append({'Metric': f'Class {i} F1 Score', 'Value': f"{class_f1_mean[i]:.4f}+/-{class_f1_std[i]:.4f}"})
            excel_data.append({'Metric': f'Class {i} F0.5 Score', 'Value': f"{class_f0_5_mean[i]:.4f}+/-{class_f0_5_std[i]:.4f}"})
            excel_data.append({'Metric': f'Class {i} F2 Score', 'Value': f"{class_f2_mean[i]:.4f}+/-{class_f2_std[i]:.4f}"})
            excel_data.append({'Metric': f'Class {i} Brier Score', 'Value': f"{class_brier_mean[i]:.4f}+/-{class_brier_std[i]:.4f}"})

        df_results = pd.DataFrame(excel_data)
        excel_filename = f'results/results_{model_name}.xlsx'
        df_results.to_excel(excel_filename, index=False)
        print(f"\nResults saved to: {excel_filename}")

        all_preds = []
        all_labels = []
        for r in results:
            all_preds.extend(r['y_pred'])
            all_labels.extend(r['y_test'])

        print(f"\n{model_name} Overall Classification Report (all folds merged):")
        print(classification_report(all_labels, all_preds,
                                   target_names=[f'Class {i}' for i in range(4)],
                                   digits=4))

        print(f"\nPlotting confusion matrix for {model_name}...")
        plot_confusion_matrix_for_model(
            all_labels=np.array(all_labels),
            all_preds=np.array(all_preds),
            model_name=model_name,
            class_names=['Distinction', 'Pass', 'Fail', 'Withdrawn'],
            save_dir='results',
            n_classes=4
        )


if __name__ == "__main__":
    excel_path = "final_data_processed.xlsx"
    X, y = load_data(excel_path)

    RANDOM_SEED = 42

    print(f"\nUsing random seed: {RANDOM_SEED}")

    all_results = cross_validate_models(X, y, n_splits=10, random_seed=RANDOM_SEED)

    print_summary(all_results)

    print("\n" + "="*50)
    print("Baseline model training completed!")
    print("="*50)
