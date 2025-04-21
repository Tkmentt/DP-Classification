import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from utils.utils import load_subjects_features, save_figure, save_model, save_classification_report, get_all_subjects_folders
from utils import config as cfg
from utils.utils_plot import plot_confusion_matrix, plot_cv_accuracy, plot_cv_loss
from preprocessing.prep_mne import ensure_preprocessed_subjects

# === Cross-validated training ===
def cross_validate_classifier(X, y, n_splits=5, random_state=42, output_dir=cfg.OUTPUT_DIR):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores, train_losses, val_losses = [], [], []
    best_acc = -1
    best_model = None
    best_scaler = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
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

        # Confusion matrix
        confusion_matrix = plot_confusion_matrix(y_val, y_pred, class_labels=[0, 1], title=f"Fold {fold} Confusion Matrix")
        save_figure(confusion_matrix, os.path.join(output_dir, f"confusion_matrix_fold{fold}.png"))
        # Report
        report = classification_report(y_val, y_pred, digits=3)
        print(report)
        save_classification_report(report, fold)

        print(f"✅ Fold {fold} — Accuracy: {acc:.3f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot accuracy per fold
    cross_val_acc = plot_cv_accuracy(acc_scores)
    save_figure(cross_val_acc, os.path.join(output_dir, "cv_accuracy_plot.png"))

    cross_val_loss = plot_cv_loss(train_losses, val_losses)
    save_figure(cross_val_loss, os.path.join(output_dir, "cv_loss_plot.png"))

    save_model(best_model, best_scaler)   

    return acc_scores, train_losses, val_losses

# Run
# === Example Batch Processing ===
if __name__ == "__main__":

    print("🔍 Checking for preprocessed subjects...")
    subject_folders = get_all_subjects_folders(cfg.DATA_DIR)
    ensure_preprocessed_subjects(subject_folders)

    print("🔍 Loading preprocessed dataset from folder")
    features, labels, subjects = load_subjects_features()
    print(f"✅ Loaded {len(features)} samples from {len(np.unique(subjects))} subjects")

    print("Starting cross-validation...")
    acc_scores, train_losses, val_losses = cross_validate_classifier(features, labels)
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)

    print(f"✅Training completed! - Accuracy: {mean_acc * 100:.2f}% ±: {std_acc * 100:.2f}%")