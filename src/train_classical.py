import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os
import datetime
from preprocessing.preprocessing import parse_marker_file, generate_sliding_windows, extract_features, compute_csp
from preprocessing.preprocessing import compute_csp, extract_features
from preprocessing.preprocessing import balance_classes
from sklearn.utils.class_weight import compute_class_weight


# === Create results folder if it does not exist ===
os.makedirs('results_classical', exist_ok=True)

def print_class_balance(labels, title="Class balance"):
    counts = np.bincount(labels)
    total = len(labels)
    print(f"{title}:")
    for idx, count in enumerate(counts):
        percentage = 100 * count / total
        print(f"  Class {idx}: {count} samples ({percentage:.2f}%)")
    print("")


# === Load full dataset with group IDs ===
def load_full_dataset(data_folder):
    all_windows = []
    all_labels = []
    all_group_ids = []
    group_counter = 0

    print(f"\n🔍 Loading full dataset from folder: {data_folder}")

    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        data_file = os.path.join(subject_path, 'data.txt')
        label_file = os.path.join(subject_path, 'labels.txt')

        if not os.path.exists(data_file) or not os.path.exists(label_file):
            print(f"⚠️ Skipping {subject_folder} (missing data or label file)")
            continue

        print(f"✅ Loading subject: {subject_folder}")

        # Load data
        data = np.genfromtxt(data_file, delimiter=',', comments='%')
        data = data[~np.isnan(data[:, -3])]  # Remove NaN timestamps

        eeg_data = data[:, [3, 4, 5]]  # Cz, C3, C4
        timestamps_raw = data[:, -3]
        timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps_raw]

        # Parse intervals
        intervals = parse_marker_file(label_file)

        # Generate sliding windows and labels
        windows, labels = generate_sliding_windows(eeg_data, timestamps, intervals)
        print_class_balance(labels, "Class balance before balancing")

        windows, labels = balance_classes(windows, labels, method='oversample')
        print_class_balance(labels, "Class balance after balancing")

        # Track group IDs (per subject)
        group_ids = np.full(len(labels), group_counter)

        all_windows.append(windows)
        all_labels.append(labels)
        all_group_ids.append(group_ids)

        group_counter += 1

    # Concatenate all data
    combined_windows = np.vstack(all_windows)
    combined_labels = np.concatenate(all_labels)
    combined_group_ids = np.concatenate(all_group_ids)

    print(f"✅ Total EEG windows: {combined_windows.shape[0]}")
    print(f"✅ Total groups (patients): {group_counter}")

    return combined_windows, combined_labels, combined_group_ids

# === Load data ===
windows, labels, group_ids = load_full_dataset('data')

# === Compute CSP filters ===
print("Computing CSP filters...")
csp_filters = compute_csp(windows, labels, n_components=4)
np.save('results_classical/csp_filters.npy', csp_filters)
print("CSP filters saved to 'results_classical/csp_filters.npy'")

# === Extract features ===
print("Extracting features...")
features = extract_features(windows, csp_filters=csp_filters)
print(f"Extracted feature shape: {features.shape}")
print("Feature sample (first row):", features[0])

# === Feature scaling ===
print("Scaling features...")
scaler = StandardScaler()
features = scaler.fit_transform(features)
joblib.dump(scaler, 'results_classical/scaler.pkl')
print("Scaler saved to 'results_classical/scaler.pkl'")

# === Check scaled feature stats ===
print("Feature sample (after scaling):", features[0])

# === Prepare input for CNN ===
features_expanded = np.expand_dims(features, -1)

# === Calculate class weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Calculated class weights:", class_weight_dict)


# === Prepare cross-validation ===
gkf = GroupKFold(n_splits=3)

# === Prepare models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for model_name, model in models.items():
    print(f"\n🚀 Training model: {model_name}")
    fold_accuracies = []
   
    for fold, (train_idx, test_idx) in enumerate(gkf.split(features_expanded, labels, groups=group_ids)):
        print(f"\n🔍 Fold {fold + 1}/{gkf.get_n_splits()}")
        print_class_balance(labels[train_idx], title="Train class balance")
        print_class_balance(labels[test_idx], title="Test class balance")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # === Feature importance (only for models that support it) ===
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.title(f"Feature Importance - {model_name} Fold {fold + 1}")
            plt.bar(range(len(importances)), importances[indices])
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(f'results_classical/feature_importance_{model_name.replace(" ", "_")}_fold_{fold + 1}.png')
            plt.close()

            print(f"✅ Feature importance plot saved for {model_name} Fold {fold + 1}")

        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.title(f"Feature Importance (Coef) - {model_name} Fold {fold + 1}")
            plt.bar(range(len(importances)), importances[indices])
            plt.xlabel("Feature Index")
            plt.ylabel("Coefficient Magnitude")
            plt.tight_layout()
            plt.savefig(f'results_classical/feature_importance_{model_name.replace(" ", "_")}_fold_{fold + 1}.png')
            plt.close()

            print(f"✅ Feature importance plot saved for {model_name} Fold {fold + 1}")

        else:
            print(f"⚠️ Model {model_name} does not support feature importance plotting.")


        acc = accuracy_score(y_test, preds)
        fold_accuracies.append(acc)
        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

        print(classification_report(y_test, preds))

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} Fold {fold + 1}')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f'results_classical/confusion_matrix_{model_name.replace(" ", "_")}_fold_{fold + 1}.png')
        plt.close()

    # Plot training accuracies over folds
    plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, marker='o')
    plt.title(f'{model_name} - Accuracy Over Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'results_classical/accuracy_{model_name.replace(" ", "_")}.png')
    plt.close()

print("\n🎉 All models trained and evaluated!")
