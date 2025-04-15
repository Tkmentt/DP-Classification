import numpy as np
import datetime
import scipy.signal as signal
from scipy.stats import entropy
import numpy as np
from scipy.linalg import eigh
from sklearn.utils import resample


def preprocess_eeg(eeg_data, sfreq):
    # Notch filter at 50 Hz
    notch_freq = 50.0
    quality_factor = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs=sfreq)
    eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=0)

    # Bandpass filter between 8–30 Hz
    lowcut = 8.0
    highcut = 22.0
    b_band, a_band = signal.butter(4, [lowcut / (0.5 * sfreq), highcut / (0.5 * sfreq)], btype='band')
    eeg_data = signal.filtfilt(b_band, a_band, eeg_data, axis=0)

    return eeg_data


def print_class_balance(labels, title="Class balance"):
    counts = np.bincount(labels)
    total = len(labels)
    print(f"{title}:")
    for idx, count in enumerate(counts):
        percentage = 100 * count / total
        print(f"  Class {idx}: {count} samples ({percentage:.2f}%)")
    print("")


def balance_classes(windows, labels, method='oversample'):
    """
    Balance the dataset using under-sampling or over-sampling.

    Parameters:
    - windows: np.array, shape (samples, time_samples, channels)
    - labels: np.array, shape (samples,)
    - method: str, 'oversample', 'undersample', or None

    Returns:
    - balanced_windows: np.array
    - balanced_labels: np.array
    """

    if method not in ['oversample', 'undersample', None]:
        raise ValueError("Method must be 'oversample', 'undersample', or None")

    if method is None:
        print("ℹ️ No class balancing applied.")
        return windows, labels

    class_0_idx = np.where(labels == 0)[0]
    class_1_idx = np.where(labels == 1)[0]

    if method == 'undersample':
        n_samples = min(len(class_0_idx), len(class_1_idx))
        class_0_selected = resample(class_0_idx, replace=False, n_samples=n_samples, random_state=42)
        class_1_selected = resample(class_1_idx, replace=False, n_samples=n_samples, random_state=42)
        print(f"🔄 Under-sampling to {n_samples} samples per class.")

    elif method == 'oversample':
        n_samples = max(len(class_0_idx), len(class_1_idx))
        class_0_selected = resample(class_0_idx, replace=True, n_samples=n_samples, random_state=42)
        class_1_selected = resample(class_1_idx, replace=True, n_samples=n_samples, random_state=42)
        print(f"🔄 Over-sampling to {n_samples} samples per class.")

    # Combine and shuffle
    selected_idx = np.concatenate([class_0_selected, class_1_selected])
    np.random.shuffle(selected_idx)

    balanced_windows = windows[selected_idx]
    balanced_labels = labels[selected_idx]

    print(f"✅ Classes after balancing: {np.bincount(balanced_labels)} (classes 0, 1)")

    return balanced_windows, balanced_labels



def compute_csp(X, y, n_components=6):
    """Compute CSP filters.
    X: EEG windows, shape (samples, time, channels)
    y: labels (0 or 1)
    n_components: number of CSP components to retain
    """
    class_labels = np.unique(y)
    if len(class_labels) != 2:
        raise ValueError("CSP requires exactly 2 classes.")

    # Calculate covariance matrices for each class
    covariances = []
    for label in class_labels:
        class_data = X[y == label]
        # Compute normalized covariance for each trial
        covs = [np.cov(trial.T) / np.trace(np.cov(trial.T)) for trial in class_data]
        covariances.append(np.mean(covs, axis=0))

    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eigh(covariances[1], covariances[0] + covariances[1])
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    print(f"CSP Explained variance ratio (top components): {explained_variance_ratio[::-1][:n_components]}")

    # Sort eigenvalues and select filters
    ix = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
    W = eigenvectors[:, ix]

    # Select n_components pairs (best variance separation)
    W = W[:, :n_components]

    return W.T  # CSP filters


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
                        "label": label
                    })
                current_phase = None

    return intervals


def is_artifact_free(window, motor_threshold=70):
    """
    Check if all channels (Cz, C3, C4) in the window are below the motor threshold.
    """
    peak_to_peak = np.ptp(window, axis=0)
    channel_names = ['Cz', 'C3', 'C4']

    for i, value in enumerate(peak_to_peak):
        if value > motor_threshold:
            print(f"⚠️ Channel {channel_names[i]} exceeds threshold: {value:.2f} μV")

    return np.all(peak_to_peak < motor_threshold)




def generate_sliding_windows(eeg_data, timestamps, intervals, window_size=2, step_size=0.2, sfreq=1000):
    """
    Generate sliding windows from EEG data based on label intervals.
    Includes preprocessing, artifact rejection, and debug printing.

    Returns:
    - windows: ndarray (n_windows, window_samples, n_channels)
    - labels: ndarray (n_windows,)
    """
    print("🚿 Preprocessing EEG signal (scaling, notch + bandpass)...")
    eeg_data = preprocess_eeg(eeg_data, sfreq)
    print("✅ Preprocessing done.")

    print(f"📏 Sampling frequency: {sfreq} Hz")
    print(f"🪟 Window size: {window_size} s ({int(window_size * sfreq)} samples)")
    print(f"🚶 Step size: {step_size} s ({int(step_size * sfreq)} samples)")

    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

    windows = []
    labels = []

    total_intervals = 0
    total_windows = 0
    total_artifact_windows = 0  # <--- Artifact counter

    for interval in intervals:
        start_time = interval['start']
        end_time = interval['end']
        label = interval['label']

        try:
            start_idx = next(i for i, t in enumerate(timestamps) if t >= start_time)
            end_idx = next(i for i, t in enumerate(timestamps) if t >= end_time)
        except StopIteration:
            print(f"⚠️ Interval {start_time} → {end_time} not found in timestamps. Skipping.")
            continue

        current_idx = start_idx
        windows_in_interval = 0
        artifacts_in_interval = 0

        while current_idx + window_samples <= end_idx:
            window = eeg_data[current_idx:current_idx + window_samples]

            # === Check for artifacts ===
            if is_artifact_free(window, motor_threshold=70):
                windows.append(window)
                labels.append(label)
            else:
                total_artifact_windows += 1
                artifacts_in_interval += 1

            current_idx += step_samples
            windows_in_interval += 1

        total_intervals += 1
        total_windows += windows_in_interval

        print(f"Interval {total_intervals}: {start_time} → {end_time} | Label: {label} | Windows: {windows_in_interval} | Artifacts: {artifacts_in_interval}")

    windows = np.array(windows)
    labels = np.array(labels, dtype=int)


    print(f"\n✅ Total intervals processed: {total_intervals}")
    print(f"✅ Total windows generated (before artifact rejection): {total_windows}")
    print(f"🧹 Total windows removed due to artifacts: {total_artifact_windows}")
    if total_windows > 0:
        print(f"🧩 Percentage of windows rejected: {100 * total_artifact_windows / total_windows:.2f}%")
    print(f"✅ Final windows kept: {len(windows)}")
    print(f"✅ Class balance: {np.bincount(labels)} (classes 0, 1)")

    return windows, labels



def extract_features(windows, sfreq=1000, csp_filters=None):
    features = []

    def hjorth_parameters(signal_data):
        first_deriv = np.diff(signal_data, axis=0)
        second_deriv = np.diff(first_deriv, axis=0)

        var_zero = np.var(signal_data, axis=0)
        var_d1 = np.var(first_deriv, axis=0)
        var_d2 = np.var(second_deriv, axis=0)

        activity = var_zero
        mobility = np.sqrt(var_d1 / (var_zero + 1e-10))
        complexity = np.sqrt((var_d2 / (var_d1 + 1e-10)) / (mobility + 1e-10))

        return activity, mobility, complexity

    def calculate_entropy(signal_data):
        hist, _ = np.histogram(signal_data, bins=50, density=True)
        hist += 1e-12
        return entropy(hist, base=2)

    def compute_stft_power(signal_data, sfreq):
        f, t, Zxx = signal.stft(signal_data, fs=sfreq, nperseg=sfreq // 4, axis=0)
        power = np.abs(Zxx) ** 2

        mu_band = np.logical_and(f >= 8, f <= 13)
        beta_band = np.logical_and(f >= 13, f <= 30)

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
            csp_logvar = np.log(np.var(csp_projection, axis=1) + 1e-10)
        else:
            csp_logvar = []

        # === Bandpower (log) ===
        freqs, psd = signal.welch(window, sfreq, nperseg=sfreq // 2, axis=0)
        mu_idx = np.logical_and(freqs >= 8, freqs <= 13)
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)

        mu_power = np.mean(psd[mu_idx, :], axis=0)
        beta_power = np.mean(psd[beta_idx, :], axis=0)

        log_mu_power = np.log(mu_power + 1e-10)       # shape: (n_channels,)
        log_beta_power = np.log(beta_power + 1e-10)   # shape: (n_channels,)

        # === Hjorth parameters ===
        activity, mobility, complexity = hjorth_parameters(window)

        # === Signal variance ===
        variance = np.var(window, axis=0)

        # === Entropy ===
        entropy_vals = np.apply_along_axis(calculate_entropy, axis=0, arr=window)

        # === Time-frequency STFT ===
        mu_stft_power, beta_stft_power = compute_stft_power(window, sfreq)

        # === Riemannian covariance features ===
        riemannian_features = compute_riemannian_features(window)

        # === Concatenate all features ===
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

