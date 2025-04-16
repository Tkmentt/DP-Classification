import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# === Settings ===
DATA_DIR = 'data'
SAVE_DIR = 'debug_plots'
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 1000  # Sampling frequency in Hz
WINDOW_SIZE = 2000  # Samples (2 seconds)
CHANNELS = ['Cz', 'C3', 'C4']
CHANNEL_IDX = [3, 4, 5]  # Columns in data.txt

# === Preprocessing from your script ===
def clip_extremes(eeg_data, clip_val=100):
    return np.clip(eeg_data, -clip_val, clip_val)

def apply_car(eeg_data):
    return eeg_data - np.mean(eeg_data, axis=1, keepdims=True)

def zscore_normalize(eeg_data):
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0) + 1e-10
    return (eeg_data - mean) / std

def preprocess_eeg(eeg_data, sfreq):
    eeg_data = clip_extremes(eeg_data, clip_val=100)
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs=sfreq)
    eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=0)
    b_band, a_band = signal.butter(4, [8.0 / (0.5 * sfreq), 22.0 / (0.5 * sfreq)], btype='band')
    eeg_data = signal.filtfilt(b_band, a_band, eeg_data, axis=0)
    eeg_data = apply_car(eeg_data)
    eeg_data = zscore_normalize(eeg_data)
    return eeg_data

def process_subject(subject_folder):
    subject_path = os.path.join(DATA_DIR, subject_folder)
    data_path = os.path.join(subject_path, 'data.txt')
    if not os.path.exists(data_path):
        print(f"⚠️ Skipping {subject_folder}: missing data.txt")
        return

    print(f"\n📁 Processing subject: {subject_folder}")
    data = np.genfromtxt(data_path, delimiter=',', comments='%')
    data = data[~np.isnan(data[:, -3])]
    eeg_raw = data[:, CHANNEL_IDX]
    print(f"🔢 Raw counts: min={np.min(eeg_raw)}, max={np.max(eeg_raw)}")

    GAIN = 24
    ADC_RESOLUTION = 2**23 - 1
    V_REF = 4.5
    scale_uV = (V_REF / ADC_RESOLUTION) * 1e6 / GAIN
    eeg_raw_uV = eeg_raw * scale_uV
    eeg_proc = preprocess_eeg(eeg_raw_uV.copy(), FS)


    if eeg_proc.shape[0] < WINDOW_SIZE:
        print(f"⚠️ Too few samples ({eeg_proc.shape[0]})")
        return

    for i, name in enumerate(CHANNELS):
        ch = eeg_proc[:, i]
        print(f"📊 {name}: mean={np.mean(ch):.3e}, std={np.std(ch):.3e}, min={np.min(ch):.3e}, max={np.max(ch):.3e}")

    for i, name in enumerate(CHANNELS):
        zeros = np.sum(eeg_proc[:, i] == 0.0)
        std = np.std(eeg_proc[:, i])
        print(f"   └─ {name} zero count: {zeros}, std: {std:.2e}")
        if std < 1e-3:
            print(f"   ❌ {name} might be constant or dead (very low variance)")

    corr = np.corrcoef(eeg_proc.T)
    print("🔗 Channel correlation matrix:")
    for i in range(len(CHANNELS)):
        for j in range(len(CHANNELS)):
            print(f"    {CHANNELS[i]}–{CHANNELS[j]}: {corr[i, j]:.2f}")

    subject_save = os.path.join(SAVE_DIR, subject_folder)
    os.makedirs(subject_save, exist_ok=True)

    # Raw + Preprocessed comparison
    for idx in range(5):
        start = np.random.randint(0, eeg_proc.shape[0] - WINDOW_SIZE)
        window_raw = eeg_raw_uV[start:start + WINDOW_SIZE, :]
        window_proc = eeg_proc[start:start + WINDOW_SIZE, :]


        plt.figure(figsize=(10, 3))
        for ch in range(3):
            plt.plot(window_raw[:, ch], alpha=0.5, label=f"{CHANNELS[ch]} (raw)")
            plt.plot(window_proc[:, ch], linestyle='--', label=f"{CHANNELS[ch]} (proc)")
        plt.title(f"Sample {idx+1} — {subject_folder}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (μV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(subject_save, f"compare_sample_{idx+1}.png"))
        plt.close()



    # Raw hist (optional, to compare)
    plt.figure(figsize=(8, 4))
    for i in range(len(CHANNELS)):
        plt.hist(eeg_raw_uV[:, i], bins=100, alpha=0.5, label=CHANNELS[i])
    plt.title(f"Amplitude Distribution (Raw) — {subject_folder}")
    plt.xlabel("Amplitude (μV)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(subject_save, "amplitude_hist_raw.png"))
    plt.close()

    # Raw PSD (optional, to compare)
    for ch_idx, name in enumerate(CHANNELS):
        f, Pxx = signal.welch(eeg_raw_uV[:, ch_idx], fs=FS, nperseg=1024)
        plt.semilogy(f, Pxx, label=name)

    plt.title(f"PSD (Raw) — {subject_folder}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(0, 50)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(subject_save, "psd_raw.png"))
    plt.close()


    plt.figure(figsize=(8, 4))
    for i in range(len(CHANNELS)):
        plt.hist(eeg_proc[:, i], bins=100, alpha=0.7, label=CHANNELS[i])
    plt.title(f"Amplitude Distribution — {subject_folder}")
    plt.xlabel("Amplitude (μV)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(subject_save, "amplitude_hist.png"))
    plt.close()

    for ch_idx, name in enumerate(CHANNELS):
        f, Pxx = signal.welch(eeg_proc[:, ch_idx], fs=FS, nperseg=1024)
        plt.semilogy(f, Pxx, label=name)

    plt.title(f"PSD — {subject_folder}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(0, 50)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(subject_save, "psd.png"))
    plt.close()

# Main loop
for subject_folder in sorted(os.listdir(DATA_DIR)):
    if os.path.isdir(os.path.join(DATA_DIR, subject_folder)):
        process_subject(subject_folder)

print("\n✅ Done. Check the 'debug_plots/' folder for visualizations.")
