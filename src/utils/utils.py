import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from utils import config as cfg 


def get_subject_id(folder_path):
    return os.path.basename(folder_path.rstrip("/\\"))

def is_subject_preprocessed(subject_folder):
    subject_id = get_subject_id(subject_folder)
    expected_file = os.path.join(cfg.PREPROCESSED_DIR, f"features_{subject_id}.npy")
    return os.path.exists(expected_file)

def get_subject_data(subject_folder):
    eeg_path = os.path.join(subject_folder, cfg.DATA_FILE)
    label_path = os.path.join(subject_folder, cfg.LABEL_FILE)

    return eeg_path, label_path


def get_all_subjects_folders(base_folder):
    """
    Get all subject folders in the base folder.
    """
    return [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]


def convert_to_microvolts(eeg_raw, gain=24, adc_resolution=2**23 - 1, v_ref=4.5):
    """
    Convert raw EEG signal to microvolts based on OpenBCI parameters.

    Parameters:
    - eeg_raw: np.ndarray, raw EEG signal
    - gain: int, amplifier gain (default 24)
    - adc_resolution: int, max ADC value (default 2^23 - 1)
    - v_ref: float, reference voltage in volts (default 4.5V)

    Returns:
    - eeg in microvolts as np.ndarray
    """
    scale_uV = (v_ref / adc_resolution) * 1e6 / gain  # µV per count
    return eeg_raw * scale_uV


def save_figure(fig, filepath, dpi=150):
    """
    Save matplotlib figure to the specified file path.

    Parameters:
    - fig: matplotlib.figure.Figure object or plt.gcf()
    - filepath: str, path to save the figure
    - dpi: int, resolution in dots per inch
    """
    ensure_dir(os.path.dirname(filepath))
    print(f"💾 Saving figure to {filepath}")
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def load_eeg_from_txt(filepath):
    """
    Load EEG data from a .txt file and extract relevant channels.
    Skips rows with NaNs in EEG signal columns.
    """
    data = np.genfromtxt(filepath, delimiter=',', comments='%')
    data = data[~np.isnan(data[:, -3])]  # Remove NaNs near EEG columns
    eeg_raw = data[:, :3]
    return eeg_raw, data



def save_subject_data_raw(path, windows, labels):
    np.save(path, {"windows": windows, "labels": labels})
    print(f"💾 Caching windows to {path}")

def load_subject_data_raw(path):
    
    cached = np.load(path, allow_pickle=True).item()
    return cached["windows"], cached["labels"]

def save_subject_features(features, labels, subject_folder):
    """
    Save features and labels for a patient in a .npy file.

    Parameters:
    - features: np.ndarray, feature data
    - labels: np.ndarray, label data
    - subject_id: str, unique identifier for the subject
    - subject_folder: str, folder to save the file in
    """

    subject_id = os.path.basename(subject_folder)
    ensure_dir(cfg.PREPROCESSED_DIR)
    save_path = os.path.join(cfg.PREPROCESSED_DIR, f'features_{subject_id}.npy')
    np.save(save_path, {
        'features': features,
        'labels': labels,
        'subject_id': subject_id
    }, allow_pickle=True)

    print(f"✅ Saved features to {save_path} — shape: {features.shape}, labels: {np.unique(labels)}")

# === Load features ===
def load_subjects_features(data_folder=cfg.PREPROCESSED_DIR):
    X_all, y_all, subjects = [], [], []
    for fname in os.listdir(data_folder):
        if fname.endswith(".npy"):
            data = np.load(os.path.join(data_folder, fname), allow_pickle=True).item()
            X_all.append(data["features"])
            y_all.append(data["labels"])
            subjects += [data["subject_id"]] * len(data["labels"])
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    subjects = np.array(subjects)
    return X_all, y_all, subjects


def save_raw_model(model, model_dir=cfg.MODEL_DIR):
    
    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, "model_raw.h5")
    model.save(model_path)
    print(f"💾 Saving model to {model_path}")



def save_model(model, scaler, model_dir=cfg.MODEL_DIR):
    """
    Save model and scaler to disk using joblib inside a directory.
    Saves to: <output_dir>/model.pkl and scaler.pkl
    """
    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"💾 Saving model to {model_path}")
    print(f"💾 Saving scaler to {scaler_path}")


def load_model(model_dir):
    """
    Load model and scaler from directory.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler



def save_figures(figures, path_template):
    """
    Save a list of matplotlib figures to file using a numbered filename pattern.

    Parameters:
    - figures: list of matplotlib.figure.Figure
    - path_template: str, must contain '{i}' to be replaced by figure number
    - dpi: int, image resolution
    """
    for i, fig in enumerate(figures):
        filename = path_template.format(i=i+1)
        save_figure(fig, filename, dpi=150)


def save_text_report(log_lines, filepath):
    """
    Save a list of text blocks (strings) into a readable .txt report.

    Parameters:
    - log_lines: list of str, each representing a section for one subject
    - filepath: str, path to save the report
    """
    ensure_dir(os.path.dirname(filepath))
    print(f"💾 Saving report to {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n\n".join(log_lines))


def save_classification_report(report, fold, output_dir=cfg.OUTPUT_DIR):
    """
    Save classification report to a text file.

    Parameters:
    - report: str, classification report
    - filepath: str, path to save the report
    """
    ensure_dir(output_dir)
    report_path = os.path.join(output_dir, f"report_fold{fold}.txt")
    print(f"💾 Saving classification report to {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
