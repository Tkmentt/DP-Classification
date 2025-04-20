import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.stats as stats
from mne.time_frequency import psd_array_welch

# === Settings ===
DATA_DIR = 'dataALL'
SAVE_DIR = 'debug_plots'
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 1000  # Sampling frequency in Hz
WINDOW_SIZE = 2000  # Samples (2 seconds)

CHANNELS = ['Cz', 'C3', 'C4']
CHANNEL_IDX = [3, 4, 5]  # Columns in data.txt


def score_artifacts(eeg, subject_id="Unknown", sfreq=1000):
# Ensure shape is (samples, channels)
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T

    picks = ['Cz', 'C3', 'C4']
    if eeg.shape[1] != 3:
        raise ValueError("Expected 3 EEG channels in order: Cz, C3, C4")

    results = []
    issues = []

    # --- Channel Correlation ---
    corr = np.corrcoef(eeg.T)
    max_corr = np.max([corr[0, 1], corr[0, 2], corr[1, 2]])
    if max_corr > 0.9:
        issues.append(f"Very high inter-channel correlation (max = {max_corr:.2f})")


    print(eeg.shape)  # Should be (samples, channels)

    # --- PSD Harmonics (detect peaks every ~10 Hz) ---
    psd, f = psd_array_welch(eeg.T, sfreq=sfreq, fmin=1, fmax=60, n_fft=2048)
    peak_freqs = f[np.argmax(psd, axis=1)]
    harmonics = np.diff(peak_freqs)
    if np.allclose(harmonics, 5, atol=1):  # spacing in bins, not Hz
        issues.append("Strong harmonics every ~10 Hz (periodic noise)")

    # --- Amplitude Range ---
    for i, signal in enumerate(eeg.T):
        ptp = np.ptp(signal)
        if ptp > 100:
            issues.append(f"{picks[i]} has excessive amplitude range ({ptp:.1f} µV)")

    # --- Kurtosis & Skew ---
    for i, signal in enumerate(eeg.T):
        k = stats.kurtosis(signal)
        s = stats.skew(signal)
        if k > 6:
            issues.append(f"{picks[i]} has high kurtosis ({k:.1f})")
        if abs(s) > 1.5:
            issues.append(f"{picks[i]} has skewed distribution (skew = {s:.1f})")

    # --- High-frequency RMS vs Alpha ---
    def band_rms(signal, f_low, f_high):
        f, psd = psd_array_welch(signal[np.newaxis], sfreq=sfreq, fmin=f_low, fmax=f_high, n_fft=1024)
        return np.sqrt(np.mean(psd))

    for i, signal in enumerate(eeg.T):
        rms_alpha = band_rms(signal, 8, 12)
        rms_high = band_rms(signal, 30, 50)
        if rms_high > 2 * rms_alpha:
            issues.append(f"{picks[i]} has high-frequency power > 2x alpha band")

    # --- Final Verdict ---
    if len(issues) == 0:
        verdict = "Good"
    elif len(issues) <= 2:
        verdict = "Suspicious"
    else:
        verdict = "Bad"

    # --- Print Results ---
    print(f"\n📁 Subject: {subject_id}")
    print(f"{'✅' if verdict == 'Good' else '⚠️' if verdict == 'Suspicious' else '❌'} Artifact Score: {verdict}")
    if issues:
        print("🔍 Reasons:")
        for reason in issues:
            print(f"  - {reason}")

    return verdict, issues


def process_subject(subject_folder):
    subject_path = os.path.join(DATA_DIR, subject_folder)
    data_path = os.path.join(subject_path, 'data.txt')

    if not os.path.exists(data_path):
        print(f"⚠️ Skipping {subject_folder}: missing data.txt")
        return


    # === Create folder for plots
    subject_save = os.path.join(SAVE_DIR, subject_folder)
    os.makedirs(subject_save, exist_ok=True)


    print(f"\n📁 Processing subject: {subject_folder}")
    data = np.genfromtxt(data_path, delimiter=',', comments='%')

    # Remove NaNs and invalid timestamps
    data = data[~np.isnan(data[:, -3])]

    eeg_raw = data[:, CHANNEL_IDX]
    
    # === Raw ADC stats ===
    print(f"🔢 Raw counts: min={np.min(eeg_raw)}, max={np.max(eeg_raw)}")

    # === Convert to microvolts based on OpenBCI gain
    GAIN = 24
    ADC_RESOLUTION = 2**23 - 1
    V_REF = 4.5  # Volts
    scale_uV = (V_REF / ADC_RESOLUTION) * 1e6 / GAIN  # µV per count
    eeg = eeg_raw * scale_uV  # Convert to µV

    verdict, reasons = score_artifacts(eeg, subject_id=subject_save, sfreq=FS)

    if verdict != "Bad":
        # === Preprocessing: Filter Definitions ===
        def notch_filter(data, fs, freq=50.0, quality=30):
            b, a = signal.iirnotch(w0=freq, Q=quality, fs=fs)
            return signal.filtfilt(b, a, data, axis=0)

        def highpass_filter(data, fs, cutoff=0.5, order=4):
            b, a = signal.butter(order, cutoff / (fs / 2), btype='high')
            return signal.filtfilt(b, a, data, axis=0)

        def bandpass_filter(data, fs, low=0.5, high=30.0, order=4):
            b, a = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
            return signal.filtfilt(b, a, data, axis=0)

        # === Preprocessing pipeline
        eeg = eeg - np.mean(eeg, axis=0)  # Remove DC offset

        eeg_notch = notch_filter(eeg, FS)
        eeg_notch_hpf = highpass_filter(eeg_notch, FS)
        eeg_notch_hpf_bpf = bandpass_filter(eeg_notch_hpf, FS)

        # === Time-Domain Comparison (RAW vs Filtered vs Fully Filtered)
        for idx in range(3):
            start = np.random.randint(0, eeg.shape[0] - WINDOW_SIZE)
            raw_win = eeg[start:start + WINDOW_SIZE]
            hpf_win = eeg_notch_hpf[start:start + WINDOW_SIZE]
            bpf_win = eeg_notch_hpf_bpf[start:start + WINDOW_SIZE]

            plt.figure(figsize=(12, 6))
            for ch_idx in range(raw_win.shape[1]):
                plt.subplot(len(CHANNELS), 1, ch_idx + 1)
                plt.plot(raw_win[:, ch_idx], 'gray', alpha=0.4, label='Raw')
                plt.plot(hpf_win[:, ch_idx], 'orange', alpha=0.7, label='Notch+HPF')
                plt.plot(bpf_win[:, ch_idx], 'blue', label='Notch+HPF+BPF')
                plt.title(f"{CHANNELS[ch_idx]} — Sample {idx+1}")
                if ch_idx == len(CHANNELS) - 1:
                    plt.xlabel("Samples")
                plt.ylabel("µV")
                if ch_idx == 0:
                    plt.legend(loc='upper right')
                plt.tight_layout()
            plt.savefig(os.path.join(subject_save, f"compare_sample_{idx+1}.png"))
            plt.close()

        # === Amplitude Histogram Comparison
        for i, name in enumerate(CHANNELS):
            plt.figure(figsize=(10, 3.5))
            plt.hist(eeg[:, i], bins=100, alpha=0.4, label='Raw')
            plt.hist(eeg_notch_hpf[:, i], bins=100, alpha=0.5, label='Notch+HPF')
            plt.hist(eeg_notch_hpf_bpf[:, i], bins=100, alpha=0.5, label='Notch+HPF+BPF')
            plt.title(f"{name} Amplitude Distribution — {subject_folder}")
            plt.xlabel("Amplitude (µV)")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(subject_save, f"amplitude_hist_{name}.png"))
            plt.close()

        # === PSD Comparison
        for i, name in enumerate(CHANNELS):
            plt.figure(figsize=(10, 4))
            for data, label in zip(
                [eeg[:, i], eeg_notch_hpf[:, i], eeg_notch_hpf_bpf[:, i]],
                ['Raw', 'Notch+HPF', 'Notch+HPF+BPF']
            ):
                f, Pxx = signal.welch(data, fs=FS, nperseg=1024)
                plt.semilogy(f, Pxx, label=label)
            plt.title(f"{name} PSD — {subject_folder}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density")
            plt.xlim(0, 60)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(subject_save, f"psd_{name}.png"))
            plt.close()



        if eeg.shape[0] < WINDOW_SIZE:
            print(f"⚠️ Too few samples ({eeg.shape[0]})")
            return

        # === Print basic stats
        for i, name in enumerate(CHANNELS):
            ch = eeg[:, i]
            print(f"📊 {name}: mean={np.mean(ch):.3e}, std={np.std(ch):.3e}, min={np.min(ch):.3e}, max={np.max(ch):.3e}")

        # === Check for zero values and flat signals
        for i, name in enumerate(CHANNELS):
            zeros = np.sum(eeg[:, i] == 0.0)
            std = np.std(eeg[:, i])
            print(f"   └─ {name} zero count: {zeros}, std: {std:.2e}")
            if std < 1e-3:
                print(f"   ❌ {name} might be constant or dead (very low variance)")

        # === Channel correlation
        corr = np.corrcoef(eeg.T)
        print("🔗 Channel correlation matrix:")
        for i in range(len(CHANNELS)):
            for j in range(len(CHANNELS)):
                print(f"    {CHANNELS[i]}–{CHANNELS[j]}: {corr[i, j]:.2f}")
        

        # === Plot random raw signal samples
        for idx in range(5):
            start = np.random.randint(0, eeg.shape[0] - WINDOW_SIZE)
            window = eeg[start:start + WINDOW_SIZE, :]

            plt.figure(figsize=(10, 2.5))
            for ch_idx in range(window.shape[1]):
                plt.plot(window[:, ch_idx], label=CHANNELS[ch_idx])
            plt.title(f"Sample {idx+1} — {subject_folder}")
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(subject_save, f"raw_sample_{idx+1}.png"))
            plt.close()

        # === Plot amplitude histogram
        plt.figure(figsize=(8, 4))
        for i in range(len(CHANNELS)):
            plt.hist(eeg[:, i], bins=100, alpha=0.7, label=CHANNELS[i])
        plt.title(f"Amplitude Distribution — {subject_folder}")
        plt.xlabel("Amplitude (µV)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(subject_save, "amplitude_hist.png"))
        plt.close()

        # === Plot PSD (Power Spectral Density)
        for ch_idx, name in enumerate(CHANNELS):
            f, Pxx = signal.welch(eeg[:, ch_idx], fs=FS, nperseg=1024)
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


# === Main loop
for subject_folder in sorted(os.listdir(DATA_DIR)):
    if os.path.isdir(os.path.join(DATA_DIR, subject_folder)):
        process_subject(subject_folder)

print("\n✅ Done. Check the 'debug_plots/' folder for visualizations.")
