import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from preprocessing.prep_stack import parse_marker_file, generate_sliding_windows, extract_features
from classification.keras.CNN import build_cnn_model, get_lr_scheduler, get_early_stopping
from preprocessing.prep_stack import compute_csp, extract_features, balance_classes, print_class_balance
import datetime
import os
import joblib

# === Create results folder ===
os.makedirs('results', exist_ok=True)

# === Plotting function for history ===
def plot_history(history, fold=None):
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True)

    if fold is not None:
        plt.savefig(f'results/training_history_fold_{fold}.png')
        plt.close()
    else:
        plt.show()

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

        eeg_data = data[:, [3, 4, 5]]  # Fp1, Fp2, Cz, C3, C4
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

from scipy.signal import welch

rest_windows = windows[labels == 0][:5]  # Just a few samples
move_windows = windows[labels == 1][:5]

for i in range(3):  # C3, Cz, C4
    f, pxx_rest = welch(rest_windows[:, :, i], fs=1000, axis=1)
    f, pxx_move = welch(move_windows[:, :, i], fs=1000, axis=1)
    
    plt.figure()
    plt.semilogy(f, np.mean(pxx_rest, axis=0), label='Rest')
    plt.semilogy(f, np.mean(pxx_move, axis=0), label='Movement')
    plt.title(f'Power Spectrum Channel {i}')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/rest_move{i}.png')
    plt.close()


# === Compute CSP filters ===
print("Computing CSP filters...")
csp_filters = compute_csp(windows, labels, n_components=8)
np.save('results/csp_filters.npy', csp_filters)
print("CSP filters saved to 'results/csp_filters.npy'")

# === Extract features with CSP ===
print("Extracting advanced features with CSP...")
features = extract_features(windows, csp_filters=csp_filters)
print(f"Extracted feature shape: {features.shape}")
print("Feature sample (first row):", features[0])

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(6, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
plt.title("PCA of Extracted Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig(f'results/PCA.png')
plt.close()


# === Feature scaling ===
print("Scaling features...")
scaler = StandardScaler()
features = scaler.fit_transform(features)
joblib.dump(scaler, 'results/scaler.pkl')
print("Scaler saved to 'results/scaler.pkl'")

print("Feature sample (after scaling):", features[0])

# === Prepare input for CNN ===
features_expanded = np.expand_dims(features, -1)

# === Calculate class weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Calculated class weights:", class_weight_dict)

# === Callbacks ===
callbacks = [
    get_lr_scheduler(),
    get_early_stopping()
]


# === Group-aware cross-validation ===
gkf = GroupKFold(n_splits=3)  # Adjust splits based on number of subjects
histories = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(features_expanded, labels, groups=group_ids)):
    print(f"\n🔍 Fold {fold + 1}/{gkf.get_n_splits()}")
    print_class_balance(labels[train_idx], title="Train class balance")
    print_class_balance(labels[test_idx], title="Test class balance")

    X_train, X_test = features_expanded[train_idx], features_expanded[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # === Build CNN model ===
    model = build_cnn_model(input_shape=(features_expanded.shape[1], 1))


    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[callbacks],
        class_weight=class_weight_dict,
        verbose=1
    )
    histories.append(history)

    # Evaluate
    preds = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, preds))

    # Confusion matrix
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
print("Saving training history...")
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