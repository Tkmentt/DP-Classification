import os
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from utils.utils import load_subjects_features, save_figure, save_model, save_classification_report, get_all_subjects_folders, get_plot_theme
from utils import config as cfg
from utils.utils_plot import plot_confusion_matrix, plot_cv_accuracy, plot_cv_loss, plot_aggregated_confusion_matrix
from joblib import dump

def cross_validate_classifier(X, y, groups=None, use_group_kfold=True,
                              n_splits=cfg.KFOLD_SPLITS, random_state=42,
                              output_dir=cfg.OUTPUT_DIR, data_type="MNE"):
    """
    Cross-validated training and evaluation with consistent theming for plots.

    Parameters:
    - X: features
    - y: labels
    - groups: optional group labels for GroupKFold
    - use_group_kfold: bool, use GroupKFold if True
    - n_splits: number of CV folds
    - random_state: for reproducibility
    - output_dir: path to save plots
    - data_type: 'MNE' or 'RAW' — determines color theme
    """

    # Determine folding type and visual theme
    folding_type = "GroupKFold" if use_group_kfold and groups is not None else "StratifiedKFold"
    theme = get_plot_theme(data_type, folding_type)

    print(f"🔀 Using {folding_type} with {data_type} theme")
    kf = GroupKFold(n_splits=n_splits) if folding_type == "GroupKFold" else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split = kf.split(X, y, groups) if groups is not None and folding_type == "GroupKFold" else kf.split(X, y)

    acc_scores, train_losses, val_losses = [], [], []
    best_acc = -1
    best_model = None
    best_scaler = None

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(split, 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=500,
                            early_stopping=True, random_state=random_state)

        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_val_scaled)
        y_proba_train = clf.predict_proba(X_train_scaled)
        y_proba_val = clf.predict_proba(X_val_scaled)

        acc = accuracy_score(y_val, y_pred)
        acc_scores.append(acc)

        train_loss = log_loss(y_train, y_proba_train)
        val_loss = log_loss(y_val, y_proba_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_scaler = scaler

        # Store predictions for aggregation
        all_y_true.append(y_val)
        all_y_pred.append(y_pred)

        # Per-fold confusion matrix
        cm_fig = plot_confusion_matrix(
            y_val, y_pred, class_labels=[0, 1],
            title=f"{theme['prefix'].upper()} Fold {fold} Confusion Matrix",
            cmap=theme["cmap"]
        )
        save_figure(cm_fig, os.path.join(output_dir, f"{theme['prefix']}_confusion_matrix_fold{fold}.png"))

        # Classification report
        report = classification_report(y_val, y_pred, digits=3)
        print(report)
        save_classification_report(report, fold, output_dir=output_dir, prefix=theme["prefix"])


        print(f"✅ Fold {fold} — Accuracy: {acc:.3f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Aggregated confusion matrix (raw)
    fig_cm, _ = plot_aggregated_confusion_matrix(
        y_trues=all_y_true,
        y_preds=all_y_pred,
        class_labels=[0, 1],
        normalize=False,
        title=f"{theme['prefix'].upper()} Final Aggregated Confusion Matrix",
        cmap=theme["cmap"]
    )
    save_figure(fig_cm, os.path.join(output_dir, f"{theme['prefix']}_confusion_matrix_final_aggregated.png"))

    # Aggregated confusion matrix (normalized)
    fig_cm_norm, _ = plot_aggregated_confusion_matrix(
        y_trues=all_y_true,
        y_preds=all_y_pred,
        class_labels=[0, 1],
        normalize=True,
        title=f"{theme['prefix'].upper()} Final Aggregated Confusion Matrix (Normalized)",
        cmap=theme["cmap"]
    )
    save_figure(fig_cm_norm, os.path.join(output_dir, f"{theme['prefix']}_confusion_matrix_final_normalized.png"))

    # Accuracy and Loss plots
    acc_fig = plot_cv_accuracy(acc_scores, title=f"{theme['prefix'].upper()} Cross-Validation Accuracy per Fold", color=theme["line_color"])
    save_figure(acc_fig, os.path.join(output_dir, f"{theme['prefix']}_cv_accuracy_plot.png"))

    loss_fig = plot_cv_loss(train_losses, val_losses, title=f"{theme['prefix'].upper()} Cross-Validation Loss per Fold", color=theme["line_color"])
    save_figure(loss_fig, os.path.join(output_dir, f"{theme['prefix']}_cv_loss_plot.png"))

    # Save best model
    save_model(best_model, best_scaler)

    return acc_scores, train_losses, val_losses



# Run
if __name__ == "__main__":

    print("🔍 Checking for preprocessed subjects...")
    subject_folders = get_all_subjects_folders(cfg.DATA_DIR)
    
    from preprocessing.prep_mne import preprocess_subject, is_subject_preprocessed

    all_csp_filters = []

    print("🔍 Checking for preprocessed subjects...")
    for folder in subject_folders:
        if not is_subject_preprocessed(folder):
            print(f"⚙️ Preprocessing needed for {folder}")
            try:
                csp = preprocess_subject(folder)
                all_csp_filters.append(csp)
            except Exception as e:
                print(f"❌ Failed to preprocess {folder}: {e}")
        else:
            print(f"✅ Already preprocessed: {folder}")


    if all_csp_filters:
        avg_csp = np.mean(np.stack(all_csp_filters), axis=0)
        dump(avg_csp, os.path.join(cfg.MODEL_DIR, "csp_filters.pkl"))
        print(f"💾 Saved average CSP filters: shape {avg_csp.shape}")


    print("🔍 Loading preprocessed dataset from folder")
    features, labels, subjects = load_subjects_features()
    print(f"✅ Loaded {len(features)} samples from {len(np.unique(subjects))} subjects")

    # Set to False to use StratifiedKFold instead
    use_group_kfold = True

    print("Starting cross-validation...")
    acc_scores, train_losses, val_losses = cross_validate_classifier(features, labels, groups=subjects, use_group_kfold=use_group_kfold)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)

    print(f"✅ Training completed! - Accuracy: {mean_acc * 100:.2f}% ±: {std_acc * 100:.2f}%")