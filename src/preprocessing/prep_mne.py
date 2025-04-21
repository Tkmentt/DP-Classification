import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import mne
import datetime
from sklearn.utils import resample
from utils import config as cfg
from preprocessing.prep_core import is_artifact_free_mne, parse_marker_file, extract_features, compute_csp
from utils.utils import load_eeg_from_txt,convert_to_microvolts,save_subject_features, get_subject_data, is_subject_preprocessed

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

# === EEG to RawArray ===
def convert_to_raw(eeg_data, sfreq=cfg.FS):
    data = eeg_data.T  # shape: (n_channels, n_samples)
    info = mne.create_info(ch_names=cfg.CHANNELS, sfreq=sfreq, ch_types=cfg.CHANNEL_TYPES)
    raw = mne.io.RawArray(data, info)
    raw.set_montage(cfg.MONTAGE)
    return raw

# === Epoch Visualization ===
def visualize_raw(raw, subject_id):
    raw.plot(duration=10.0, n_channels=3, title=f"{subject_id} — Raw EEG", show=True, block=False)



# === Sliding Window Generator ===
def generate_sliding_windows(eeg_data, timestamps, intervals, window_size=2.0, step_size=cfg.STEP_SIZE, sfreq=cfg.FS):
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

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
        window_count = 0

        while current_idx + window_samples <= end_idx:
            window = eeg_data[current_idx:current_idx + window_samples, :]
            # === Artifact check here
            if not is_artifact_free_mne(window, verbose=True):
                current_idx += step_samples
                window_count += 1
                continue  # skip bad window

            label = float(raw_label)
            windows.append(window)
            labels.append(label)

            current_idx += step_samples
            window_count += 1


    return np.array(windows), np.array(labels, dtype=np.float32)




# === Main Preprocessing ===
def preprocess_subject(subject_folder):

    print(f"\n📁 Processing subject: {subject_folder}")

    eeg_path, label_path = get_subject_data(subject_folder)
    
    eeg = load_eeg_from_txt(eeg_path)
    eeg = convert_to_microvolts(eeg)
    raw = convert_to_raw(eeg)
    raw.notch_filter(cfg.NOTCH_TRESHOLD)
    raw.filter(1., cfg.HIGH_BAND_THRESHOLD)

    intervals = parse_marker_file(label_path)
    start_time = intervals[0]['start'] if intervals else datetime.datetime.now()
    timestamps = [start_time + datetime.timedelta(seconds=i/cfg.FS) for i in range(eeg.shape[0])]

    #visualize_raw(raw, os.path.basename(subject_folder))

    eeg_filtered = raw.get_data().T
    windows, labels = generate_sliding_windows(eeg_filtered, timestamps, intervals, sfreq=cfg.FS)
    windows, labels = balance_windows(windows, labels)
    
    csp_filters = compute_csp(windows, labels)
    features = extract_features(windows, sfreq=cfg.FS, csp_filters=csp_filters)

    save_subject_features(features, labels, subject_folder)


def ensure_preprocessed_subjects(subject_folders):
    """
    Check and preprocess any subjects that haven't been processed yet.
    """
    for folder in subject_folders:
        if not is_subject_preprocessed(folder):
            print(f"⚙️ Preprocessing needed for {folder}")
            try:
                preprocess_subject(folder)
            except Exception as e:
                print(f"❌ Failed to preprocess {folder}: {e}")
        else:
            print(f"✅ Already preprocessed: {folder}")


