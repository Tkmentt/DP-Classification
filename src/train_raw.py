import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing.preprocessing import parse_marker_file, generate_sliding_windows, balance_classes
from classification.keras.CNNTransformerLSTM import build_cnn_bilstm_model
import datetime
import os
import joblib

# === Create results folder if it does not exist ===
os.makedirs('results', exist_ok=True)

def print_class_balance(labels, title="Class balance"):
    counts = np.bincount(labels)
    total = len(labels)
    print(f"{title}:")
    for idx, count in enumerate(counts):
        percentage = 100 * count / total
        print(f"  Class {idx}: {count} samples ({percentage:.2f}%)")
    print("")

# === Load full raw dataset with group IDs ===
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

        group_ids = np.full(len(labels), group_counter)

        all_windows.append(windows)
        all_labels.append(labels)
        all_group_ids.append(group_ids)

        group_counter += 1

    combined_windows = np.vstack(all_windows)
    combined_labels = np.concatenate(all_labels)
    combined_group_ids = np.concatenate(all_group_ids)

    print(f"✅ Total EEG windows: {combined_windows.shape[0]}")
    print(f"✅ Total groups (patients): {group_counter}")

    return combined_windows, combined_labels, combined_group_ids

# === Load data ===
windows, labels, group_ids = load_full_dataset('data')

# === Scale raw EEG windows ===
print("Scaling raw EEG windows...")
scaler = StandardScaler()
# Flatten for scaler
windows_flat = windows.reshape(-1, windows.shape[-1])
windows_scaled = scaler.fit_transform(windows_flat)
# Reshape back
windows_scaled = windows_scaled.reshape(windows.shape)
joblib.dump(scaler, 'results/scaler.pkl')
print("Scaler saved to 'results/scaler.pkl'")

# === Prepare input shape ===
print(f"Input shape for CNN-BiLSTM: {windows_scaled.shape}")

# === Calculate class weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Calculated class weights:", class_weight_dict)

# === Build CNN-BiLSTM model ===
model = build_cnn_bilstm_model(input_shape=(windows_scaled.shape[1], windows_scaled.shape[2]))

# === Group-aware cross-validation ===
gkf = GroupKFold(n_splits=5)
histories = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(windows_scaled, labels, groups=group_ids)):
    print(f"\n🔍 Fold {fold + 1}/{gkf.get_n_splits()}")
    print_class_balance(labels[train_idx], title="Train class balance")
    print_class_balance(labels[test_idx], title="Test class balance")

    X_train, X_test = windows_scaled[train_idx], windows_scaled[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1
    )
    histories.append(history)

    preds = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Fold {fold + 1}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f'results/confusion_matrix_fold_{fold + 1}.png')
    plt.close()

# === Save final model ===
model.save('results/model.h5')
print("✅ Model saved to 'results/model.h5'")

# === Plot training history ===
plt.figure(figsize=(10, 5))
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Fold {i + 1} Accuracy')
plt.title('Training Accuracy Over Folds')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('results/training_accuracy.png')
plt.close()
print("✅ Training history saved to 'results/training_accuracy.png'")

print("\n🎉 Training completed successfully!")
