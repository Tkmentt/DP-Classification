import os
import numpy as np
import mne
import datetime
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, entropy
from scipy import signal
from scipy.linalg import eigh
from sklearn.utils import resample

# === Config ===
CHANNEL_NAMES = ['Cz', 'C3', 'C4']
CHANNEL_TYPES = ['eeg'] * 3
SFREQ = 1000
MONTAGE = 'standard_1020'
EVENT_MAPPING = {
    "S3": 0,  # Rest
    "S4": 1,  # Move
    "S5": 1,  # Move (assist)
    "S6": 0   # Relax
}


def balance_windows(windows, labels):
    class_0 = windows[labels == 0]
    class_1 = windows[labels == 1]

    if len(class_0) < len(class_1):
        class_0_upsampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
        labels_0 = np.zeros(len(class_1))
        windows_balanced = np.vstack([class_0_upsampled, class_1])
        labels_balanced = np.concatenate([labels_0, np.ones(len(class_1))])
    else:
        class_1_upsampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
        labels_1 = np.ones(len(class_0))
        windows_balanced = np.vstack([class_0, class_1_upsampled])
        labels_balanced = np.concatenate([np.zeros(len(class_0)), labels_1])

    return windows_balanced, labels_balanced


# === Marker Parser ===
def parse_marker_file(marker_file_path):
    intervals = []
    current_phase = None
    start_time = None

    with open(marker_file_path, "r") as file:
        for line in file:
            if len(line.strip()) == 0:
                continue
            timestamp_str, marker = line.strip().split(" ", 1)
            timestamp = datetime.datetime.strptime(timestamp_str, "%y.%m.%d-%H:%M:%S.%f")
            if marker.startswith("S"):
                current_phase = marker
                start_time = timestamp
            elif marker.startswith("R") and current_phase:
                label = EVENT_MAPPING.get(current_phase, None)
                if label is not None:
                    intervals.append({
                        "start": start_time,
                        "end": timestamp,
                        "label": label,
                        "phase": current_phase
                    })
                current_phase = None
    return intervals

# === EEG Loader ===
def load_eeg_txt(file_path):
    raw_data = np.genfromtxt(file_path, delimiter=',', comments='%')
    
    # Remove rows with NaNs in timestamp column (assumed last 3 columns are timestamps or meta)
    raw_data = raw_data[~np.isnan(raw_data[:, -3])]
    
    # Extract EEG channels Cz, C3, C4 (first 3 columns)
    eeg_data = raw_data[:, :3]

    # === Convert to microvolts based on OpenBCI gain
    GAIN = 24  # OpenBCI default gain
    ADC_RESOLUTION = 2**23 - 1  # 24-bit ADS1299
    V_REF = 4.5  # Reference voltage in volts
    scale_uV = (V_REF / ADC_RESOLUTION) * 1e6 / GAIN  # µV per count
    eeg_data = eeg_data * scale_uV  # Convert ADC counts to microvolts

    return eeg_data


# === EEG to RawArray ===
def convert_to_raw(eeg_data, sfreq=SFREQ):
    data = eeg_data.T  # shape: (n_channels, n_samples)
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types=CHANNEL_TYPES)
    raw = mne.io.RawArray(data, info)
    raw.set_montage(MONTAGE)
    return raw

# === CSP Computation ===
def compute_csp(windows, labels, n_components=6, reg=1e-6):
    class0 = windows[labels == 0]
    class1 = windows[labels == 1]
    cov0 = np.mean([np.cov(w.T) for w in class0], axis=0)
    cov1 = np.mean([np.cov(w.T) for w in class1], axis=0)
    cov0 += reg * np.eye(cov0.shape[0])
    cov1 += reg * np.eye(cov1.shape[0])
    composite = cov0 + cov1
    eigenvalues, eigenvectors = eigh(cov1, composite)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    print(f"📊 CSP Explained variance ratio: {explained_variance_ratio[::-1][:n_components]}")
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors[:, :n_components]

# === Full Feature Extractor ===
def extract_features(windows, labels, sfreq=1000, csp_filters=None):

    features = []
    for window in windows:
        if csp_filters is not None:
            csp_projection = np.dot(csp_filters, window.T)
            csp_logvar = np.log(np.var(csp_projection, axis=1) + 1e-10)
        else:
            csp_logvar = []

        freqs, psd = signal.welch(window, sfreq, nperseg=sfreq // 2, axis=0)
        mu_idx = np.logical_and(freqs >= 8, freqs <= 13)
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)

        mu_power = np.mean(psd[mu_idx, :], axis=0)
        beta_power = np.mean(psd[beta_idx, :], axis=0)

        log_mu_power = np.log(mu_power + 1e-10)
        log_beta_power = np.log(beta_power + 1e-10)

        first_deriv = np.diff(window, axis=0)
        second_deriv = np.diff(first_deriv, axis=0)

        var_zero = np.var(window, axis=0)
        var_d1 = np.var(first_deriv, axis=0)
        var_d2 = np.var(second_deriv, axis=0)

        mobility = np.sqrt(var_d1 / (var_zero + 1e-10))
        complexity = np.sqrt((var_d2 / (var_d1 + 1e-10)) / (mobility + 1e-10))

        hist_entropy = np.apply_along_axis(
            lambda x: entropy(np.histogram(x, bins=50, density=True)[0] + 1e-12, base=2),
            axis=0, arr=window)

        f, t, Zxx = signal.stft(window, fs=sfreq, nperseg=sfreq // 4, axis=0)
        power = np.abs(Zxx) ** 2
        mu_band = np.logical_and(f >= 8, f <= 13)
        beta_band = np.logical_and(f >= 13, f <= 30)
        mu_stft_power = np.mean(power[mu_band, :, :], axis=(0, 1, 2))
        beta_stft_power = np.mean(power[beta_band, :, :], axis=(0, 1, 2))

        cov_matrix = np.cov(window, rowvar=False) + 1e-6 * np.eye(window.shape[1])
        log_eigenvalues = np.log(np.maximum(np.linalg.eigvalsh(cov_matrix), 1e-6))

        feature_vector = np.concatenate([
            csp_logvar,
            log_mu_power,
            log_beta_power,
            var_zero,
            mobility,
            complexity,
            hist_entropy,
            [mu_stft_power],
            [beta_stft_power],
            log_eigenvalues
        ])
        features.append(feature_vector)
    return np.array(features)


def is_artifact_free(window, motor_threshold=100, min_std=0.1, max_z=10, verbose=False):
    """
    Real-time-safe artifact rejection:
    - Peak-to-peak (amplitude burst)
    - Flatline (very low std)
    - Z-score based threshold (excessive variance)
    """
    channel_names = ['Cz', 'C3', 'C4']
    peak_to_peak = np.ptp(window, axis=0)
    stds = np.std(window, axis=0)

    if np.any(stds < min_std):
        if verbose: print("⚠️ Flatline detected.")
        return False

    if np.any(peak_to_peak > motor_threshold):
        if verbose:
            for i, val in enumerate(peak_to_peak):
                if val > motor_threshold:
                    print(f"⚠️ Channel {channel_names[i]} exceeds threshold: {val:.2f} µV")
        return False

    # Optional z-score vs median variance
    var = np.var(window, axis=0)
    z = (var - np.median(var)) / (np.std(var) + 1e-10)
    if np.any(np.abs(z) > max_z):
        if verbose: print("⚠️ High z-score variance.")
        return False

    return True


# === Sliding Window Generator ===
def generate_sliding_windows(eeg_data, timestamps, intervals, window_size=2.0, step_size=0.2, sfreq=1000):
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    decay_s5 = 1
    decay_s6 = 1

    windows = []
    labels = []

    for interval in intervals:
        start_time = interval['start']
        end_time = interval['end']
        raw_label = interval['label']
        marker = interval.get('phase', None)

        try:
            start_idx = next(i for i, t in enumerate(timestamps) if t >= start_time)
            end_idx = next(i for i, t in enumerate(timestamps) if t >= end_time)
        except StopIteration:
            continue

        current_idx = start_idx
        #interval_duration_sec = (end_idx - start_idx) / sfreq
        #decay_limit_samples = int(4.0 / step_size)

        window_count = 0

        while current_idx + window_samples <= end_idx:
            window = eeg_data[current_idx:current_idx + window_samples, :]
            # === Artifact check here
            if not is_artifact_free(window, verbose=True):
                current_idx += step_samples
                window_count += 1
                continue  # skip bad window

            label = float(raw_label)
            windows.append(window)
            labels.append(label)

            current_idx += step_samples
            window_count += 1


    return np.array(windows), np.array(labels, dtype=np.float32)

# === Epoch Visualization ===
def visualize_raw(raw, subject_id):
    raw.plot(duration=10.0, n_channels=3, title=f"{subject_id} — Raw EEG", show=True, block=False)

# === Main Preprocessing ===
def preprocess_subject(subject_folder):
    print(f"\n📁 Processing subject: {subject_folder}")
    eeg_path = os.path.join(subject_folder, 'data.txt')
    label_path = os.path.join(subject_folder, 'labels.txt')

    eeg = load_eeg_txt(eeg_path)
    raw = convert_to_raw(eeg)
    raw.notch_filter(50)
    raw.filter(1., 30.)

    intervals = parse_marker_file(label_path)
    start_time = intervals[0]['start'] if intervals else datetime.datetime.now()
    timestamps = [start_time + datetime.timedelta(seconds=i/SFREQ) for i in range(eeg.shape[0])]

    #visualize_raw(raw, os.path.basename(subject_folder))

    eeg_filtered = raw.get_data().T
    windows, labels = generate_sliding_windows(eeg_filtered, timestamps, intervals, sfreq=SFREQ)
    csp_filters = compute_csp(windows, labels)
    features = extract_features(windows, labels, sfreq=SFREQ, csp_filters=csp_filters)

    subject_id = os.path.basename(subject_folder)
    os.makedirs('preprocessed_data', exist_ok=True)
    save_path = os.path.join('preprocessed_data', f'features_{subject_id}.npy')
    np.save(save_path, {
        'features': features,
        'labels': labels,
        'subject_id': subject_id
    }, allow_pickle=True)
    print(f"✅ Saved features to {save_path} — shape: {features.shape}, labels: {np.unique(labels)}")

# === Example Batch Processing ===
if __name__ == "__main__":
    root = "dataALL"  # 🔁 Replace this
    subject_folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for folder in subject_folders:
        try:
            preprocess_subject(folder)
        except Exception as e:
            print(f"❌ Failed to process {folder}: {e}")
