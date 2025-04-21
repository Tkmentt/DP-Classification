import numpy as np
from scipy.stats import entropy
from scipy import signal
from sklearn.utils import resample
from scipy.linalg import eigh
import datetime
from utils import config as cfg

# === Basic Preprocessing Utilities ===

def clip_extremes(eeg_data, clip_val=100):
    return np.clip(eeg_data, -clip_val, clip_val)

def apply_car(eeg_data):
    return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

def zscore_normalize(eeg_data):
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0) + 1e-10
    return (eeg_data - mean) / std

def highpass_filter(eeg_data, sfreq, cutoff=0.5, order=4):
    b, a = signal.butter(order, cutoff / (0.5 * sfreq), btype='high')
    return signal.filtfilt(b, a, eeg_data, axis=0)

# === Marker Parsing ===

def parse_marker_file(marker_file_path):
    intervals = []
    current_phase = None
    start_time = None

    event_mapping = {
        "S6": 0,
        "S3": 0,
        "S4": 1,
        "S5": 1
    }

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
                label = event_mapping.get(current_phase, None)
                if label is not None:
                    intervals.append({
                        "start": start_time,
                        "end": timestamp,
                        "label": label,
                        "phase": current_phase
                    })
                current_phase = None

    return intervals


def is_artifact_free_mne(window, motor_threshold=100, min_std=0.1, max_z=10, verbose=False):  
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
def extract_features(windows, sfreq=cfg.FS, csp_filters=None):
    features = []

    def hjorth_parameters(signal_data):
        first_deriv = np.diff(signal_data, axis=0)
        second_deriv = np.diff(first_deriv, axis=0)

        var_zero = np.var(signal_data, axis=0)
        var_d1 = np.var(first_deriv, axis=0)
        var_d2 = np.var(second_deriv, axis=0)

        activity = var_zero
        mobility = np.sqrt(var_d1 / (var_zero + cfg.EPSILON))
        complexity = np.sqrt((var_d2 / (var_d1 + cfg.EPSILON)) / (mobility + cfg.EPSILON))
        return activity, mobility, complexity

    def calculate_entropy(signal_data):
        hist, _ = np.histogram(signal_data, bins=50, density=True)
        hist += cfg.EPSILON
        return entropy(hist, base=2)

    def compute_stft_power(signal_data, sfreq):
        f, t, Zxx = signal.stft(signal_data, fs=sfreq, nperseg=sfreq // 4, axis=0)
        power = np.abs(Zxx) ** 2

        mu_band = np.logical_and(f >= cfg.MU_LOW, f <= cfg.MU_HIGH)
        beta_band = np.logical_and(f >= cfg.BETA_LOW, f <= cfg.BETA_HIGH)

        mu_power = np.mean(power[mu_band, :, :], axis=(0, 1, 2))
        beta_power = np.mean(power[beta_band, :, :], axis=(0, 1, 2))

        return mu_power, beta_power

    def compute_riemannian_features(signal_data):
        cov_matrix = np.cov(signal_data, rowvar=False)
        cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        log_eigenvalues = np.log(eigenvalues)
        return log_eigenvalues

    for window in windows:
        # === CSP features ===
        if csp_filters is not None:
            csp_projection = np.dot(csp_filters, window.T)
            csp_logvar = np.log(np.var(csp_projection, axis=1) + cfg.EPSILON)
        else:
            csp_logvar = []

        # === Bandpower (log) ===
        freqs, psd = signal.welch(window, sfreq, nperseg=sfreq // 2, axis=0)
        mu_idx = np.logical_and(freqs >= cfg.MU_LOW, freqs <= cfg.MU_HIGH)
        beta_idx = np.logical_and(freqs >= cfg.BETA_LOW, freqs <= cfg.BETA_HIGH)

        mu_power = np.mean(psd[mu_idx, :], axis=0)
        beta_power = np.mean(psd[beta_idx, :], axis=0)

        log_mu_power = np.log(mu_power + cfg.EPSILON)
        log_beta_power = np.log(beta_power + cfg.EPSILON)

        # === Hjorth ===
        activity, mobility, complexity = hjorth_parameters(window)

        # === Signal variance ===
        variance = np.var(window, axis=0)

        # === Entropy ===
        entropy_vals = np.apply_along_axis(calculate_entropy, axis=0, arr=window)

        # === Time-frequency STFT power ===
        mu_stft_power, beta_stft_power = compute_stft_power(window, sfreq)

        # === Riemannian covariance features ===
        riemannian_features = compute_riemannian_features(window)

        # === Combine all features ===
        feature_vector = np.concatenate([
            csp_logvar,
            log_mu_power,
            log_beta_power,
            activity,
            mobility,
            complexity,
            variance,
            entropy_vals,
            [mu_stft_power],
            [beta_stft_power],
            riemannian_features
        ])

        features.append(feature_vector)

    return np.array(features)





# === NOT USED IN MNE VERSIONS ===


def is_artifact_free(window, motor_threshold=100):
    """
    Check if all channels (Cz, C3, C4) in the window are below the motor threshold.
    """
    peak_to_peak = np.ptp(window, axis=0)
    channel_names = ['Cz', 'C3', 'C4']

    for i, value in enumerate(peak_to_peak):
        if value > motor_threshold:
            print(f"⚠️ Channel {channel_names[i]} exceeds threshold: {value:.2f} μV")

    return np.all(peak_to_peak < motor_threshold)


def preprocess_eeg_for_cnn(eeg_data_uV, sfreq):
    """
    Minimal preprocessing for raw EEG CNNs:
    - High-pass filter (0.5 Hz)
    - Common average referencing
    """
    # 1. High-pass to remove slow drift
    eeg_data = highpass_filter(eeg_data_uV, sfreq, cutoff=0.5)

    # 2. Apply CAR
    eeg_data = apply_car(eeg_data)

    # ⚠️ Do NOT apply z-score or bandpass here

    return eeg_data


def preprocess_eeg(eeg_data, sfreq):
    # Step 1: Clip big spikes (optional)
    eeg_data = clip_extremes(eeg_data, clip_val=100)

    # Step 2: Notch filter at 50 Hz
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs=sfreq)
    eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=0)

    # Step 3: Bandpass filter 8–22 Hz
    b_band, a_band = signal.butter(4, [8.0 / (0.5 * sfreq), 22.0 / (0.5 * sfreq)], btype='band')
    eeg_data = signal.filtfilt(b_band, a_band, eeg_data, axis=0)

    # Step 4: Common average reference (CAR)
    eeg_data = apply_car(eeg_data)

    # Step 5: Z-score normalization (optional)
    eeg_data = zscore_normalize(eeg_data)

    return eeg_data



def generate_sliding_windows_decay(eeg_data, timestamps, intervals, raw=False, window_size=2, step_size=0.2, sfreq=1000):
    
    """
    Generate sliding windows from EEG data with decayed labels for S5 and S6.
    """

    if raw:
        eeg_data = preprocess_eeg_for_cnn(eeg_data, sfreq)
    else:
        eeg_data = preprocess_eeg(eeg_data, sfreq)

    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

    windows = []
    labels = []

    decay_s5 = 1  # After 4 seconds, S5 becomes 0.5
    decay_s6 = 0  # First 4 seconds of S6 are 0.5, rest are 0

    for interval in intervals:
        start_time = interval['start']
        end_time = interval['end']
        raw_label = interval['label']
        marker = interval.get('phase', None)  # get 'S5' or 'S6'

        try:
            start_idx = next(i for i, t in enumerate(timestamps) if t >= start_time)
            end_idx = next(i for i, t in enumerate(timestamps) if t >= end_time)
        except StopIteration:
            continue

        current_idx = start_idx
        interval_duration_sec = (end_idx - start_idx) / sfreq
        decay_limit_samples = int(4.0 / step_size)

        window_count = 0

        while current_idx + window_samples <= end_idx:
            window = eeg_data[current_idx:current_idx + window_samples]

            if is_artifact_free(window):
                label = float(raw_label)  # S3/S4: just use standard binary 0/1
                windows.append(window)
                labels.append(label)

            current_idx += step_samples
            window_count += 1

    windows = np.array(windows)
    labels = np.array(labels, dtype=np.float32)

    print(f"✅ Total windows kept: {len(windows)}")
    print(f"✅ Label stats — min: {labels.min()}, max: {labels.max()}, unique: {np.unique(labels)}")

    return windows, labels


def generate_dual_windows(eeg_data, timestamps, intervals, window_size=2, step_size=0.2, sfreq=1000):
    """
    Generates both raw and heavily preprocessed EEG windows from the same source for hybrid models.
    Returns: windows_raw (for CNN), windows_features (for MLP), labels
    """
    # Light preprocessing for raw CNN input
    eeg_raw = preprocess_eeg_for_cnn(eeg_data.copy(), sfreq)

    # Full preprocessing for MLP feature extraction (CAR, clip, z-score)
    eeg_feat = preprocess_eeg(eeg_data.copy(), sfreq)

    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

    windows_raw = []
    windows_feat = []
    labels = []

    decay_s5 = 0.5
    decay_s6 = 0.5

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
        decay_limit_samples = int(4.0 / step_size)
        window_count = 0

        while current_idx + window_samples <= end_idx:
            win_raw = eeg_raw[current_idx:current_idx + window_samples]
            win_feat = eeg_feat[current_idx:current_idx + window_samples]

            # ✅ Artifact check on both signals
            if is_artifact_free(win_feat) and is_artifact_free(win_raw):
                if marker == 'S5':
                    label = 1.0 if window_count < decay_limit_samples else decay_s5
                elif marker == 'S6':
                    label = decay_s6 if window_count < decay_limit_samples else 0.0
                else:
                    label = float(raw_label)

                windows_raw.append(win_raw)
                windows_feat.append(win_feat)
                labels.append(label)

            current_idx += step_samples
            window_count += 1

    return np.array(windows_raw), np.array(windows_feat), np.array(labels, dtype=np.float32)


# === Class Handling ===

def print_class_balance(labels, title="Class balance"):
    labels = np.array(labels)
    print(f"{title}:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for label, count in zip(unique_labels, counts):
        percentage = 100 * count / total
        print(f"  Class {label:.1f}: {count} samples ({percentage:.2f}%)")
    print("")

def balance_classes(windows, labels, method='oversample'):
    if method not in ['oversample', 'undersample', None]:
        raise ValueError("Method must be 'oversample', 'undersample', or None")

    if method is None:
        print("ℹ️ No class balancing applied.")
        return windows, labels

    hard_0_idx = np.where(labels == 0.0)[0]
    hard_1_idx = np.where(labels == 1.0)[0]
    soft_idx = np.where((labels != 0.0) & (labels != 1.0))[0]

    if method == 'undersample':
        n_samples = min(len(hard_0_idx), len(hard_1_idx))
        hard_0_selected = resample(hard_0_idx, replace=False, n_samples=n_samples, random_state=42)
        hard_1_selected = resample(hard_1_idx, replace=False, n_samples=n_samples, random_state=42)
        print(f"🔄 Under-sampling to {n_samples} samples per hard class.")
    elif method == 'oversample':
        n_samples = max(len(hard_0_idx), len(hard_1_idx))
        hard_0_selected = resample(hard_0_idx, replace=True, n_samples=n_samples, random_state=42)
        hard_1_selected = resample(hard_1_idx, replace=True, n_samples=n_samples, random_state=42)
        print(f"🔄 Over-sampling to {n_samples} samples per hard class.")

    selected_idx = np.concatenate([hard_0_selected, hard_1_selected, soft_idx])
    np.random.shuffle(selected_idx)

    balanced_windows = windows[selected_idx]
    balanced_labels = labels[selected_idx]

    print(f"✅ After balancing (with soft labels):")
    print(f"  Class 0.0: {np.sum(balanced_labels == 0.0)}")
    print(f"  Class 1.0: {np.sum(balanced_labels == 1.0)}")
    print(f"  Soft labels (0.5): {np.sum(balanced_labels == 0.5)}")

    return balanced_windows, balanced_labels
# === Class Handling ===

def print_class_balance(labels, title="Class balance"):
    labels = np.array(labels)
    print(f"{title}:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    for label, count in zip(unique_labels, counts):
        percentage = 100 * count / total
        print(f"  Class {label:.1f}: {count} samples ({percentage:.2f}%)")
    print("")

def balance_classes(windows, labels, method='oversample'):
    if method not in ['oversample', 'undersample', None]:
        raise ValueError("Method must be 'oversample', 'undersample', or None")

    if method is None:
        print("ℹ️ No class balancing applied.")
        return windows, labels

    hard_0_idx = np.where(labels == 0.0)[0]
    hard_1_idx = np.where(labels == 1.0)[0]
    soft_idx = np.where((labels != 0.0) & (labels != 1.0))[0]

    if method == 'undersample':
        n_samples = min(len(hard_0_idx), len(hard_1_idx))
        hard_0_selected = resample(hard_0_idx, replace=False, n_samples=n_samples, random_state=42)
        hard_1_selected = resample(hard_1_idx, replace=False, n_samples=n_samples, random_state=42)
        print(f"🔄 Under-sampling to {n_samples} samples per hard class.")
    elif method == 'oversample':
        n_samples = max(len(hard_0_idx), len(hard_1_idx))
        hard_0_selected = resample(hard_0_idx, replace=True, n_samples=n_samples, random_state=42)
        hard_1_selected = resample(hard_1_idx, replace=True, n_samples=n_samples, random_state=42)
        print(f"🔄 Over-sampling to {n_samples} samples per hard class.")

    selected_idx = np.concatenate([hard_0_selected, hard_1_selected, soft_idx])
    np.random.shuffle(selected_idx)

    balanced_windows = windows[selected_idx]
    balanced_labels = labels[selected_idx]

    print(f"✅ After balancing (with soft labels):")
    print(f"  Class 0.0: {np.sum(balanced_labels == 0.0)}")
    print(f"  Class 1.0: {np.sum(balanced_labels == 1.0)}")
    print(f"  Soft labels (0.5): {np.sum(balanced_labels == 0.5)}")

    return balanced_windows, balanced_labels



def balance_classes_dual(labels, return_indices=False):
    """
    Oversample hard-labeled (0 and 1) classes to balance them.
    Keep soft-labeled (0.5) samples untouched.
    """
    
    labels = np.array(labels)
    idx_0 = np.where(labels == 0.0)[0]
    idx_1 = np.where(labels == 1.0)[0]
    idx_soft = np.where(labels == 0.5)[0]

    n0, n1 = len(idx_0), len(idx_1)
    max_hard = max(n0, n1)

    # Oversample the smaller hard class
    if n0 < max_hard:
        idx_0_upsampled = np.random.choice(idx_0, size=max_hard, replace=True)
    else:
        idx_0_upsampled = idx_0

    if n1 < max_hard:
        idx_1_upsampled = np.random.choice(idx_1, size=max_hard, replace=True)
    else:
        idx_1_upsampled = idx_1

    # Combine hard-balanced + all soft labels
    balanced_idx = np.concatenate([idx_0_upsampled, idx_1_upsampled, idx_soft])
    np.random.shuffle(balanced_idx)

    if return_indices:
        return balanced_idx
    else:
        return labels[balanced_idx]