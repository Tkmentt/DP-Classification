import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from preprocessing.prep_core import parse_marker_file, generate_dual_windows
from classification.keras.CNN import build_cnn_model, get_lr_scheduler, get_early_stopping
from classification.keras.MLP import build_mlp_model
from preprocessing.prep_core import compute_csp, extract_features, balance_classes, print_class_balance
import datetime
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Create results folder ===
os.makedirs('results', exist_ok=True)

# === Load full dataset with group IDs ===
def load_full_dataset(data_folder):
    all_raw_windows, all_feat_windows, all_labels, all_group_ids = [], [], [], []
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

                # Generate dual window views: raw (for CNN) and heavily preprocessed (for MLP)
        raw_windows, feature_windows, labels = generate_dual_windows(eeg_data, timestamps, intervals)

        # Check balance before
        print_class_balance(labels, "Class balance before balancing")
        print("Label summary before balancing:")
        print("Unique labels:", np.unique(labels))
        print("Soft label count (0.5):", np.sum(labels == 0.5))
        windows, labels = balance_classes(windows, labels, method='oversample')

        # === Balance using consistent indices
        balanced_idx = balance_classes(labels, return_indices=True)

        raw_windows = raw_windows[balanced_idx]
        feature_windows = feature_windows[balanced_idx]
        labels = labels[balanced_idx]
        group_ids = np.full(len(labels), group_counter)

        # Check balance after
        print_class_balance(labels, "Class balance after balancing")
        print("Label summary after balancing:")
        print("Unique labels:", np.unique(labels))
        print("Soft label count (0.5):", np.sum(labels == 0.5))

        group_ids = np.full(len(labels), group_counter)
        group_counter += 1


        # Collect
        all_raw_windows.append(raw_windows)
        all_feat_windows.append(feature_windows)
        all_labels.append(labels)
        all_group_ids.append(group_ids)
        group_counter += 1

    # Concatenate everything
    combined_raw = np.vstack(all_raw_windows)
    combined_feat = np.vstack(all_feat_windows)
    combined_labels = np.concatenate(all_labels)
    combined_group_ids = np.concatenate(all_group_ids)

    print(f"\n✅ Total EEG windows: {combined_raw.shape[0]}")
    print(f"✅ Total groups (patients): {group_counter}")

    return combined_raw, combined_feat, combined_labels, combined_group_ids




################################################################################################

# === Load the raw windows and labels ===
windows, labels, group_ids = load_full_dataset('data')


# === Filter for hard labels before CSP ===
hard_label_mask = (labels == 0) | (labels == 1)
windows_hard = windows[hard_label_mask]
labels_hard = labels[hard_label_mask].astype(int)
# === Compute CSP filters only on hard-labeled data ===
print("\n🧠 Computing CSP filters...")
csp_filters = compute_csp(windows_hard, labels_hard, n_components=6)

# === Extract features from CLEANED RAW ===
print("🧪 Extracting feature vectors...")
features = extract_features(windows, csp_filters=csp_filters)

print(f"✅ Feature shape: {features.shape}")
print(f"🔍 Example feature vector: {features[0]}")


# === Prepare outer GroupKFold ===
n_outer_folds = 3
gkf = GroupKFold(n_splits=n_outer_folds)
folds = list(gkf.split(windows, labels, groups=group_ids))

meta_features_train = []
meta_labels_train = []

# Outer fold loop
for fold_idx, (train_idx, test_idx) in enumerate(folds):
    print(f"\n🔍 Outer Fold {fold_idx + 1}/{n_outer_folds}")

    # Split raw & feature data
    X_raw_train, X_raw_test = windows[train_idx], windows[test_idx]
    X_feat_train, X_feat_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # One-hot encode for soft labels
    y_train_cat = np.stack([1 - y_train, y_train], axis=1)
    y_test_cat = np.stack([1 - y_test, y_test], axis=1)

    # Standard scaling
    scaler = StandardScaler()
    X_feat_train_scaled = scaler.fit_transform(X_feat_train)
    X_feat_test_scaled = scaler.transform(X_feat_test)

    # === INNER K-Fold for meta-training ===
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for inner_train_idx, inner_val_idx in skf.split(X_feat_train, (y_train == 1).astype(int)):

        # Inner split
        X_raw_inner_train = X_raw_train[inner_train_idx]
        X_feat_inner_train = X_feat_train_scaled[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        y_inner_train_cat = np.stack([1 - y_inner_train, y_inner_train], axis=1)

        X_raw_inner_val = X_raw_train[inner_val_idx]
        X_feat_inner_val = X_feat_train_scaled[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]

        # CNN
        cnn = build_cnn_model(input_shape=X_raw_inner_train.shape[1:])
        cnn.fit(X_raw_inner_train, y_inner_train_cat,
                epochs=30, batch_size=32, verbose=0,
                validation_split=0.1,
                callbacks=[get_lr_scheduler(), get_early_stopping()])
        cnn_probs_val = cnn.predict(X_raw_inner_val)

        # MLP
        mlp = build_mlp_model(input_shape=X_feat_inner_train.shape[1])
        mlp.fit(X_feat_inner_train, y_inner_train_cat,
                epochs=30, batch_size=32, verbose=0,
                validation_split=0.1,
                callbacks=[get_lr_scheduler(), get_early_stopping()])
        mlp_probs_val = mlp.predict(X_feat_inner_val)

        # Append to meta training data
        meta_features_train.append(np.hstack([cnn_probs_val, mlp_probs_val]))
        meta_labels_train.append(y_inner_val)

    # === Final models trained on full outer train set
    cnn_final = build_cnn_model(input_shape=X_raw_train.shape[1:])
    cnn_final.fit(X_raw_train, y_train_cat,
                  epochs=50, batch_size=32, verbose=0,
                  validation_split=0.1,
                  callbacks=[get_lr_scheduler(), get_early_stopping()])
    cnn_probs_test = cnn_final.predict(X_raw_test)

    mlp_final = build_mlp_model(input_shape=X_feat_train_scaled.shape[1])
    mlp_final.fit(X_feat_train_scaled, y_train_cat,
                  epochs=50, batch_size=32, verbose=0,
                  validation_split=0.1,
                  callbacks=[get_lr_scheduler(), get_early_stopping()])
    mlp_probs_test = mlp_final.predict(X_feat_test_scaled)

    # Store final test fold predictions
    fold_meta = np.hstack([cnn_probs_test, mlp_probs_test])
    if fold_idx == 0:
        all_test_meta = fold_meta
        all_test_labels = y_test
    else:
        all_test_meta = np.vstack([all_test_meta, fold_meta])
        all_test_labels = np.concatenate([all_test_labels, y_test])

# === Train meta-classifier on meta training set ===
X_meta_train = np.vstack(meta_features_train)
y_meta_train = np.concatenate(meta_labels_train)

meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
meta_clf.fit(X_meta_train, y_meta_train.astype(int))  # Must convert to int!

# === Evaluate on all held-out test folds (only hard labels)
hard_mask = (all_test_labels == 0) | (all_test_labels == 1)
true_hard_labels = all_test_labels[hard_mask].astype(int)
preds_hard = meta_clf.predict(all_test_meta[hard_mask])

print("\n📊 Final Meta-Classifier Evaluation:")
print(classification_report(true_hard_labels, preds_hard, digits=3))

cm = confusion_matrix(true_hard_labels, preds_hard)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Final Stacked Confusion Matrix (All Folds)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('results/final_stacked_confusion_matrix.png')
plt.close()

