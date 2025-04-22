import os
import numpy as np
import datetime
from utils import config as cfg
from utils.utils import convert_to_microvolts, get_all_subjects_folders, ensure_dir, load_eeg_from_txt, \
                        get_subject_data, load_subject_data_raw, save_subject_data_raw
from preprocessing.prep_core import parse_marker_file, generate_sliding_windows_decay, balance_classes, print_class_balance

def load_full_dataset(data_folder=cfg.DATA_DIR, cache_dir=cfg.PREPROCESSED_RAW_DIR):
    ensure_dir(cache_dir)

    all_windows, all_labels, all_group_ids = [], [], []
    group_counter = 0

    print(f"\n🔍 Loading full dataset from folder: {data_folder}")

    subject_folders = get_all_subjects_folders(data_folder)
    for subject_path in subject_folders:
        subject_id = os.path.basename(subject_path)
        cache_file = os.path.join(cache_dir, f"raw_windows_{subject_id}.npy")

        if os.path.exists(cache_file):
            print(f"📦 Loading cached windows for {subject_id}")
            windows, labels = load_subject_data_raw(cache_file)
        else:
            data_file, label_file = get_subject_data(subject_path)

            if not os.path.exists(data_file) or not os.path.exists(label_file):
                print(f"⚠️ Skipping {subject_id} (missing files)")
                continue

            print(f"✅ Processing {subject_id}...")

            eeg_data , data = load_eeg_from_txt(data_file)
            eeg_data = convert_to_microvolts(eeg_data)

            timestamps_raw = data[:, -3]
            timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps_raw]
            intervals = parse_marker_file(label_file)

            windows, labels = generate_sliding_windows_decay(eeg_data, timestamps, intervals, raw=True)
            print_class_balance(labels, "Before balancing")

            windows, labels = balance_classes(windows, labels, method='oversample')
            print_class_balance(labels, "After balancing")

            # Cache it
            save_subject_data_raw(cache_file, windows, labels)
            print(f"💾 Cached windows to {cache_file}")

        group_ids = np.full(len(labels), group_counter)
        group_counter += 1

        all_windows.append(windows)
        all_labels.append(labels)
        all_group_ids.append(group_ids)

    combined_windows = np.vstack(all_windows)
    combined_labels = np.concatenate(all_labels)
    combined_group_ids = np.concatenate(all_group_ids)

    print(f"\n✅ Total EEG windows: {combined_windows.shape[0]}")
    print(f"✅ Total groups (patients): {group_counter}")

    return combined_windows, combined_labels, combined_group_ids
