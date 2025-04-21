import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy import signal
import scipy.stats as stats
from mne.time_frequency import psd_array_welch
from utils.utils import convert_to_microvolts, save_figure, save_figures, ensure_dir, load_eeg_from_txt, save_text_report
from utils import config as config 
from utils import utils_plot

CORRELATION_THRESHOLD = 0.9  # Threshold for high correlation between channels
AMPLITUDE_THRESHOLD = 100  # Threshold for excessive amplitude range in µV
KURTOSIS_THRESHOLD = 6  # Threshold for high kurtosis
SKEW_THRESHOLD = 1.5  # Threshold for skewness
ALPHA_BAND = (8, 12)  # Alpha band frequency range in Hz
HIGH_BAND = (30, 50)  # High-frequency band range in Hz
GOOD_THRESHOLD = 0  # Threshold for good signal quality
SUSPICIOUS_THRESHOLD = 3  # Threshold for suspicious signal quality

subject_score_log = []


def score_artifacts(eeg, subject_id="Unknown", sfreq=config.FS):
# Ensure shape is (samples, channels)
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T

    picks = config.CHANNELS
    if eeg.shape[1] != 3:
        raise ValueError("Expected 3 EEG channels in order: Cz, C3, C4")

    results = []
    issues = []

    # --- Channel Correlation ---
    corr = np.corrcoef(eeg.T)
    max_corr = np.max([corr[0, 1], corr[0, 2], corr[1, 2]])
    if max_corr > CORRELATION_THRESHOLD:
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
        if ptp > AMPLITUDE_THRESHOLD:
            issues.append(f"{picks[i]} has excessive amplitude range ({ptp:.1f} µV)")

    # --- Kurtosis & Skew ---
    for i, signal in enumerate(eeg.T):
        k = stats.kurtosis(signal)
        s = stats.skew(signal)
        if k > KURTOSIS_THRESHOLD:
            issues.append(f"{picks[i]} has high kurtosis ({k:.1f})")
        if abs(s) > SKEW_THRESHOLD:
            issues.append(f"{picks[i]} has skewed distribution (skew = {s:.1f})")

    # --- High-frequency RMS vs Alpha ---
    def band_rms(signal, f_low, f_high):
        f, psd = psd_array_welch(signal[np.newaxis], sfreq=sfreq, fmin=f_low, fmax=f_high, n_fft=1024)
        return np.sqrt(np.mean(psd))

    for i, signal in enumerate(eeg.T):
        rms_alpha = band_rms(signal, *ALPHA_BAND)
        rms_high = band_rms(signal,  *HIGH_BAND)
        if rms_high > 2 * rms_alpha:
            issues.append(f"{picks[i]} has high-frequency power > 2x alpha band")

    # --- Final Verdict ---
    if len(issues) == GOOD_THRESHOLD:
        verdict = "Good"
    elif len(issues) <= SUSPICIOUS_THRESHOLD:
        verdict = "Suspicious"
    else:
        verdict = "Bad"

    # --- Print Results ---
    print(f"\n📁 Subject: {subject_id}")
    print(f"{'✅' if verdict == 'Good' else '⚠️' if verdict == 'Suspicious' else '❌'} Artifact Score: {verdict}")
    log_lines = [f"\n📁 Subject: {subject_id}",
             f"{'✅' if verdict == 'Good' else '⚠️' if verdict == 'Suspicious' else '❌'} Artifact Score: {verdict}"]
    
    if issues:
        print("🔍 Reasons:")
        log_lines.append("🔍 Reasons:")
        for reason in issues:
            log_lines.append(f"  - {reason}")
            print(f"  - {reason}")

    subject_score_log.append("\n".join(log_lines))
    return verdict, issues


def process_subject(subject_folder):
    subject_path = os.path.join(config.DATA_DIR, subject_folder)
    data_path = os.path.join(subject_path, config.DATA_FILE)

    if not os.path.exists(data_path):
        print(f"⚠️ Skipping {subject_folder}: missing data.txt")
        return


    # === Create folder for plots
    subject_save = os.path.join(config.DEBUG_DIR, subject_folder)
    ensure_dir(subject_save)
    ensure_dir(config.DEBUG_DIR)

    print(f"\n📁 Processing subject: {subject_folder}")
    eeg_raw = load_eeg_from_txt(data_path, channel_idx=config.CHANNEL_IDX)

    # === Raw ADC stats ===
    print(f"🔢 Raw counts: min={np.min(eeg_raw)}, max={np.max(eeg_raw)}")

    # === Convert to microvolts based on OpenBCI gain
    eeg = convert_to_microvolts(eeg_raw)

    verdict, reasons = score_artifacts(eeg, subject_id=subject_save)

    if verdict == "Dont":
        # === Preprocessing: Filter Definitions ===
        def notch_filter(data, fs, freq=config.NOTCH_TRESHOLD, quality=30):
            b, a = signal.iirnotch(w0=freq, Q=quality, fs=fs)
            return signal.filtfilt(b, a, data, axis=0)

        def highpass_filter(data, fs, cutoff=config.LOW_BAND_THRESHOLD, order=4):
            b, a = signal.butter(order, cutoff / (fs / 2), btype='high')
            return signal.filtfilt(b, a, data, axis=0)

        def bandpass_filter(data, fs, low=config.LOW_BAND_THRESHOLD, high=config.HIGH_BAND_THRESHOLD, order=4):
            b, a = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
            return signal.filtfilt(b, a, data, axis=0)

        # === Preprocessing pipeline
        eeg = eeg - np.mean(eeg, axis=0)  # Remove DC offset
        eeg_notch = notch_filter(eeg, config.FS)
        eeg_notch_hpf = highpass_filter(eeg_notch, config.FS)
        eeg_notch_hpf_bpf = bandpass_filter(eeg_notch_hpf, config.FS)

        # === Time-Domain Comparison (RAW vs Filtered vs Fully Filtered)
        time_domain_comparison = utils_plot.plot_time_domain_comparison(
            eeg, [eeg_notch, eeg_notch_hpf, eeg_notch_hpf_bpf], ['Raw', 'Notch', 'Notch+HPF', 'Notch+HPF+BPF'],
            subject_title=subject_folder
        )
        save_figures(time_domain_comparison, os.path.join(subject_save, 'time_domain_comparison_{i}.png'))

        # === Amplitude Histogram Comparison
        amplitude_comparison = utils_plot.plot_amplitude_histogram_comparison_per_channel(
            eeg, [eeg_notch_hpf, eeg_notch_hpf_bpf], ['Raw', 'Notch+HPF', 'Notch+HPF+BPF'],
            subject_title=subject_folder
        )        
        save_figures(amplitude_comparison, os.path.join(subject_save, 'amplitude_histogram_comparison_{i}.png'))

        # === PSD Comparison
        psd_comparison = utils_plot.plot_psd_comparison_per_channel(
            eeg, [eeg_notch_hpf, eeg_notch_hpf_bpf], ['Raw', 'Notch+HPF', 'Notch+HPF+BPF'],
            subject_title=subject_folder, xlim=(0, 60)
        )
        save_figures(psd_comparison, os.path.join(subject_save, 'psd_comparison_{i}.png'))


        if eeg.shape[0] < config.WINDOW_SIZE:
            print(f"⚠️ Too few samples ({eeg.shape[0]})")
            return

        # === Print basic stats
        for i, name in enumerate(config.CHANNELS):
            ch = eeg[:, i]
            print(f"📊 {name}: mean={np.mean(ch):.3e}, std={np.std(ch):.3e}, min={np.min(ch):.3e}, max={np.max(ch):.3e}")

        # === Check for zero values and flat signals
        for i, name in enumerate(config.CHANNELS):
            zeros = np.sum(eeg[:, i] == 0.0)
            std = np.std(eeg[:, i])
            print(f"   └─ {name} zero count: {zeros}, std: {std:.2e}")
            if std < 1e-3:
                print(f"   ❌ {name} might be constant or dead (very low variance)")

        # === Channel correlation
        corr = np.corrcoef(eeg.T)
        print("🔗 Channel correlation matrix:")
        for i in range(len(config.CHANNELS)):
            for j in range(len(config.CHANNELS)):
                print(f"    {config.CHANNELS[i]}–{config.CHANNELS[j]}: {corr[i, j]:.2f}")
        

        # === Plot random raw signal samples
        random_raw_windows = utils_plot.plot_random_raw_windows(
            eeg, subject_title=subject_folder, n_windows=5
        )
        save_figures(random_raw_windows, os.path.join(subject_save, 'random_raw_windows_{i}.png'))

        # === Plot amplitude histogram
        amplitude_hist = utils_plot.plot_amplitude_histogram(
            eeg, subject_title=subject_folder
        )
        save_figure(amplitude_hist, os.path.join(subject_save, 'amplitude_histogram.png'))

        # === Plot PSD (Power Spectral Density)
        psd_plot = utils_plot.plot_multichannel_psd(
            eeg, subject_title=subject_folder, xlim=(0, 60)
        )
        save_figure(psd_plot, os.path.join(subject_save, 'psd_plot.png'))


# === Main loop
for subject_folder in sorted(os.listdir(config.DATA_DIR)):
    subject_path = os.path.join(config.DATA_DIR, subject_folder)
    if os.path.isdir(subject_path):
        process_subject(subject_folder)

save_text_report(subject_score_log, os.path.join(config.DEBUG_DIR, config.SCORE_REPORT_FILE))
print(f"\n✅ Done. Check the {config.DEBUG_DIR} folder for visualizations.")
