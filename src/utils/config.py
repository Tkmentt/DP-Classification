# === Global Configuration ===

FS = 1000  # Sampling frequency in Hz
WINDOW_SIZE = 2000  # Samples (2 seconds)
STEP_SIZE = 0.2  # Seconds (200 ms)

TESTING_STEP_SIZE = 0.3 # Seconds (300 ms)
KFOLD_SPLITS = 5  # Number of splits for k-fold
SMOOTHING_WINDOW = 3  # Number of samples for smoothing

CHANNELS = ['Cz', 'C3', 'C4']
CHANNEL_IDX = [3, 4, 5]  # Columns in data.txt
CHANNEL_IDX_TESTING = [2, 3, 4]
CHANNEL_TYPES = ['eeg'] * 3
MONTAGE = 'standard_1020'
EVENT_MAPPING = {
    "S3": 0,  # Rest
    "S4": 1,  # Move
    "S5": 1,  # Move (assist)
    "S6": 0   # Relax
}


# Band definitions
NOTCH_TRESHOLD = 50.0 # Notch filter frequency in Hz
LOW_BAND_THRESHOLD = 8.0  # Low bandpass filter frequency in Hz
HIGH_BAND_THRESHOLD = 22.0  # High bandpass filter frequency in Hz
MU_LOW = 8.0  # Mu band low frequency in Hz
MU_HIGH = 13.0  # Mu band high frequency in Hz
BETA_LOW = 13.0  # Beta band low frequency in Hz
BETA_HIGH = 30.0  # Beta band high frequency in Hz
EPSILON = 1e-10  # Small value to avoid division by zero

DATA_FILE = 'data.txt'  # Data file name
LABEL_FILE = 'labels.txt'  # Label file name
SCORE_REPORT_FILE = 'score_report.txt'  # Score report file name

# Directory defaults
DATA_DIR = 'dataALL'
PREPROCESSED_DIR = 'preprocessed_data'
PREPROCESSED_RAW_DIR = 'preprocessed_raw_data'
DEBUG_DIR = 'debug_plots'
OUTPUT_DIR = "outputs"
OUTPUT_RAW_DIR = "outputs_raw"
MODEL_DIR = "models"
TESTING_LOG_DIR = "logs"
