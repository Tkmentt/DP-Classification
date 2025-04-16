import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing.preprocessing_better import parse_marker_file, generate_sliding_windows,  balance_classes, print_class_balance
from classification.keras.CNN import build_cnn_model, get_lr_scheduler, get_early_stopping
import datetime
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Create results folder ===
os.makedirs('resultsRAW', exist_ok=True)

# === Load full dataset with group IDs ===
def load_full_dataset(data_folder):
    all_windows, all_labels, all_group_ids = [], [], []
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

        data = np.genfromtxt(data_file, delimiter=',', comments='%')
        data = data[~np.isnan(data[:, -3])]  # Remove NaN timestamps

        eeg_data = data[:, [3, 4, 5]]  # Cz, C3, C4 (raw ADC counts)
        # === Convert to microvolts based on OpenBCI gain
        GAIN = 24
        ADC_RESOLUTION = 2**23 - 1
        V_REF = 4.5  # Volts
        scale_uV = (V_REF / ADC_RESOLUTION) * 1e6 / GAIN  # µV per count
        eeg_data = eeg_data * scale_uV  # Convert to µV
        
        timestamps_raw = data[:, -3]
        timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps_raw]

        intervals = parse_marker_file(label_file)

        windows, labels = generate_sliding_windows(eeg_data, timestamps, intervals)
        print_class_balance(labels, "Class balance before balancing")

        print("Label summary before balancing:")
        print("Unique labels:", np.unique(labels))
        print("Soft label count (0.5):", np.sum(labels == 0.5))
        windows, labels = balance_classes(windows, labels, method='oversample')
        print_class_balance(labels, "Class balance after balancing")
        print("Label summary after balancing:")
        print("Unique labels:", np.unique(labels))
        print("Soft label count (0.5):", np.sum(labels == 0.5))

        group_ids = np.full(len(labels), group_counter)
        group_counter += 1

        all_windows.append(windows)
        all_labels.append(labels)
        all_group_ids.append(group_ids)

    combined_windows = np.vstack(all_windows)
    combined_labels = np.concatenate(all_labels)
    combined_group_ids = np.concatenate(all_group_ids)

    print(f"✅ Total EEG windows: {combined_windows.shape[0]}")
    print(f"✅ Total groups (patients): {group_counter}")

    return combined_windows, combined_labels, combined_group_ids



# === Load raw windows and labels ===
windows, labels, group_ids = load_full_dataset('dataALL')

# === Prepare GroupKFold ===
n_splits = 3
gkf = GroupKFold(n_splits=n_splits)
folds = list(gkf.split(windows, labels, groups=group_ids))
print(f"\n🔁 Prepared {n_splits}-fold GroupKFold splits.")

# === Iterate through folds ===
for fold_idx, (train_idx, test_idx) in enumerate(folds):
    print(f"\n🔍 Fold {fold_idx + 1}/{n_splits}")

    # Split raw EEG
    X_train_raw = windows[train_idx]
    X_test_raw = windows[test_idx]

    # Split labels
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # Convert soft labels to one-hot
    y_train_cat = np.stack([1 - y_train, y_train], axis=1)
    y_test_cat = np.stack([1 - y_test, y_test], axis=1)

    # Build and train CNN
    cnn_model = build_cnn_model(input_shape=(X_train_raw.shape[1], X_train_raw.shape[2]))
    cnn_model.fit(
        X_train_raw, y_train_cat,
        validation_data=(X_test_raw, y_test_cat),
        epochs=50,
        batch_size=32,
        callbacks=[get_lr_scheduler(), get_early_stopping()],
        verbose=1
    )

    # Predict and evaluate only on hard labels
    cnn_probs = cnn_model.predict(X_test_raw)
    cnn_preds = np.argmax(cnn_probs, axis=1)

    hard_mask = (y_test == 0) | (y_test == 1)
    y_test_hard = y_test[hard_mask].astype(int)
    cnn_preds_hard = cnn_preds[hard_mask]

    print("🧠 CNN Validation Results (Hard Labels Only):")
    print(classification_report(y_test_hard, cnn_preds_hard, digits=3))

    cm = confusion_matrix(y_test_hard, cnn_preds_hard)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CNN Confusion Matrix Fold {fold_idx + 1}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f'resultsRAW/cnn_confusion_matrix_fold_{fold_idx + 1}.png')
    plt.close()
