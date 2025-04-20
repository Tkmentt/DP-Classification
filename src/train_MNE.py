import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import joblib

# === Load features ===
def load_all_preprocessed_features(data_folder="preprocessed_data"):
    X_all, y_all, subjects = [], [], []
    for fname in os.listdir(data_folder):
        if fname.endswith(".npy"):
            data = np.load(os.path.join(data_folder, fname), allow_pickle=True).item()
            X_all.append(data["features"])
            y_all.append(data["labels"])
            subjects += [data["subject_id"]] * len(data["labels"])
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    subjects = np.array(subjects)
    return X_all, y_all, subjects

# === Cross-validated training ===
def cross_validate_classifier(X, y, n_splits=5, random_state=42, output_dir="cv_results"):
    os.makedirs(output_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores, train_losses, val_losses = [], [], []

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

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.title(f"Fold {fold} — Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_fold{fold}.png"))
        plt.close()

        # Report
        report = classification_report(y_val, y_pred, digits=3)
        print(report)
        with open(os.path.join(output_dir, f"report_fold{fold}.txt"), "w") as f:
            f.write(report)

        print(f"✅ Fold {fold} — Accuracy: {acc:.3f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot accuracy per fold
    plt.figure(figsize=(6, 4))
    plt.plot(acc_scores, marker='o', label="Validation Accuracy")
    plt.axhline(np.mean(acc_scores), linestyle='--', color='gray', label='Mean Accuracy')
    plt.title("Cross-Validation Accuracy per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_accuracy_plot.png"))
    plt.close()

    # Plot loss per fold
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, marker='o', label="Train Loss")
    plt.plot(val_losses, marker='s', label="Val Loss")
    plt.title("Cross-Validation Loss per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_loss_plot.png"))
    plt.close()

    return acc_scores, train_losses, val_losses

# Run
print(f"\n🔍 Loading preprocessed dataset from folder")
X, y, subjects = load_all_preprocessed_features()
print(f"✅ Loaded {len(X)} samples from {len(np.unique(subjects))} subjects")
print("Starting cross-validation...")
acc_scores, train_losses, val_losses = cross_validate_classifier(X, y)
mean_acc = np.mean(acc_scores)
std_acc = np.std(acc_scores)

print(f"Training completed! - Accuracy: {mean_acc * 100:.2f}% ±: {std_acc * 100:.2f}%")

