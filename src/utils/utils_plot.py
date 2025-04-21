import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.signal import welch
from utils import config

def plot_confusion_matrix(y_true, y_pred, class_labels, title="Confusion Matrix", normalize=False, cmap="Blues"):
    
    print(f"📈Plotting {title}")
    
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_cv_loss(train_losses, val_losses, title="Cross-Validation Loss per Fold"):
    """
    Plot training and validation loss per fold.

    Parameters:
    - train_losses: list of float
    - val_losses: list of float
    - title: plot title

    Returns:
    - matplotlib Figure object
    """
    print(f"📈Plotting {title}")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, marker='o', label="Train Loss")
    plt.plot(val_losses, marker='s', label="Val Loss")
    plt.title(title)
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_cv_accuracy(acc_scores, title="Cross-Validation Accuracy per Fold"):
    """
    Plot accuracy values per fold for cross-validation.

    Parameters:
    - acc_scores: list or np.ndarray of accuracy values (0–1)
    - title: plot title

    Returns:
    - matplotlib Figure object
    """
    print(f"📈Plotting {title}")

    plt.figure(figsize=(6, 4))
    plt.plot(acc_scores, marker='o', label="Validation Accuracy")
    plt.axhline(np.mean(acc_scores), linestyle='--', color='gray', label='Mean Accuracy')
    plt.title(title)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_raw_eeg(signal, fs=config.FS, title="EEG Signal", channel_names=config.CHANNELS):

    print(f"📈Plotting {title}")

    plt.figure(figsize=(12, 6))
    time = np.arange(signal.shape[1]) / fs
    for i, channel in enumerate(signal):
        offset = i * np.max(np.abs(signal)) * 1.5
        plt.plot(time, channel + offset, label=channel_names[i] if channel_names else f"Ch{i+1}")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_multichannel_psd(eeg, fs=config.FS, channel_names=config.CHANNELS, subject_title="PSD Plot", xlim=(0, 50)):
    """
    Plot the Power Spectral Density (PSD) for each EEG channel.

    Parameters:
    - eeg: np.ndarray, shape (samples, channels)
    - fs: int, sampling frequency
    - channel_names: list of str, names of channels
    - subject_title: str, plot title
    - xlim: tuple, x-axis frequency limits

    Returns:
    - matplotlib Figure object
    """

    print(f"📈Plotting PSD — {subject_title}")

    plt.figure(figsize=(10, 5))
    for ch_idx, name in enumerate(channel_names):
        f, Pxx = welch(eeg[:, ch_idx], fs=fs, nperseg=1024)
        plt.semilogy(f, Pxx, label=name)

    plt.title(f"PSD — {subject_title}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(*xlim)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_amplitude_histogram(eeg, channel_names=config.CHANNELS, subject_title="Amplitude Distribution"):
    """
    Plot amplitude histogram for each EEG channel.

    Parameters:
    - eeg: np.ndarray, shape (samples, channels)
    - channel_names: list of str, names of channels
    - subject_title: str, plot title

    Returns:
    - matplotlib Figure object
    """
    print(f"Amplitude Distribution — {subject_title}")
    
    plt.figure(figsize=(8, 4))
    for i in range(len(channel_names)):
        plt.hist(eeg[:, i], bins=100, alpha=0.7, label=channel_names[i])
    plt.title(f"Amplitude Distribution — {subject_title}")
    plt.xlabel("Amplitude (µV)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_random_raw_windows(eeg, channel_names=config.CHANNELS, subject_title="Raw EEG Sample", window_size=config.WINDOW_SIZE, n_windows=5):
    """
    Plot several randomly selected raw EEG windows.

    Parameters:
    - eeg: np.ndarray, shape (samples, channels)
    - channel_names: list of str, names of channels
    - subject_title: str, base title for each plot
    - window_size: int, number of samples in each window
    - n_windows: int, number of windows to plot

    Returns:
    - List of matplotlib Figure objects
    """
    print(f"📈Plotting {n_windows} random raw EEG windows — {subject_title}")
    
    figures = []
    for idx in range(n_windows):
        start = np.random.randint(0, eeg.shape[0] - window_size)
        window = eeg[start:start + window_size, :]

        plt.figure(figsize=(10, 2.5))
        for ch_idx in range(window.shape[1]):
            plt.plot(window[:, ch_idx], label=channel_names[ch_idx])
        plt.title(f"Sample {idx+1} — {subject_title}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.tight_layout()
        figures.append(plt.gcf())
    return figures


def plot_psd_comparison_per_channel(raw_eeg, filtered_eegs, labels, fs=config.FS, channel_names=config.CHANNELS, subject_title="", xlim=(0, 60)):
    """
    Plot PSD comparisons (e.g. raw vs. filtered) for each EEG channel.

    Parameters:
    - raw_eeg: np.ndarray, shape (samples, channels)
    - filtered_eegs: list of np.ndarray, each of shape (samples, channels)
    - labels: list of str, labels for the plots (e.g., ['Raw', 'Notch+HPF', 'Notch+HPF+BPF'])
    - fs: int, sampling frequency
    - channel_names: list of str, names of EEG channels
    - subject_title: str, prefix for plot title
    - xlim: tuple, x-axis frequency limits

    Returns:
    - List of matplotlib Figure objects
    """
    print(f"📈Plotting PSD Comparison — {subject_title}")

    figures = []
    for i, name in enumerate(channel_names):
        plt.figure(figsize=(10, 4))
        for eeg_data, label in zip([raw_eeg[:, i]] + [f[:, i] for f in filtered_eegs], labels):
            f, Pxx = welch(eeg_data, fs=fs, nperseg=1024)
            plt.semilogy(f, Pxx, label=label)
        plt.title(f"{name} PSD — {subject_title}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.xlim(*xlim)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        figures.append(plt.gcf())
    return figures



def plot_amplitude_histogram_comparison_per_channel(raw_eeg, filtered_eegs, labels, channel_names=config.CHANNELS, subject_title="Amplitude Comparison"):
    """
    Plot amplitude histogram comparisons (e.g. raw vs. filtered) for each EEG channel.

    Parameters:
    - raw_eeg: np.ndarray, shape (samples, channels)
    - filtered_eegs: list of np.ndarray, each of shape (samples, channels)
    - labels: list of str, e.g. ['Raw', 'Notch+HPF', 'Notch+HPF+BPF']
    - channel_names: list of str, names of EEG channels
    - subject_title: str, title suffix

    Returns:
    - List of matplotlib Figure objects
    """
    print(f"📈Plotting Amplitude Histogram Comparison — {subject_title}")

    figures = []
    for i, name in enumerate(channel_names):
        plt.figure(figsize=(10, 3.5))
        plt.hist(raw_eeg[:, i], bins=100, alpha=0.4, label=labels[0])
        for j, f in enumerate(filtered_eegs):
            plt.hist(f[:, i], bins=100, alpha=0.5, label=labels[j + 1])
        plt.title(f"{name} Amplitude Distribution — {subject_title}")
        plt.xlabel("Amplitude (µV)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        figures.append(plt.gcf())
    return figures


def plot_time_domain_comparison(raw_eeg, filtered_eegs, labels, channel_names=config.CHANNELS, window_size=config.WINDOW_SIZE, n_windows=3, subject_title="Time-Domain Comparison"):
    """
    Plot time-domain EEG comparison across filtering stages.

    Parameters:
    - raw_eeg: np.ndarray, shape (samples, channels)
    - filtered_eegs: list of np.ndarray, each (samples, channels)
    - labels: list of str, names for each filter stage (Raw, HPF, BPF, ...)
    - channel_names: list of str
    - window_size: int, number of samples in each window
    - n_windows: int, number of random samples
    - subject_title: str, suffix for figure title

    Returns:
    - List of matplotlib Figure objects
    """

    print(f"📈Plotting Time-Domain Comparison — {subject_title}")

    figures = []
    for idx in range(n_windows):
        start = np.random.randint(0, raw_eeg.shape[0] - window_size)
        windows = [raw_eeg[start:start + window_size]] + [f[start:start + window_size] for f in filtered_eegs]

        plt.figure(figsize=(12, 6))
        for ch_idx in range(windows[0].shape[1]):
            plt.subplot(len(channel_names), 1, ch_idx + 1)
            for win, label, color, alpha in zip(windows, labels, ['gray', 'orange', 'blue'], [0.4, 0.7, 1.0]):
                plt.plot(win[:, ch_idx], label=label, alpha=alpha, color=color)
            plt.title(f"{channel_names[ch_idx]} — Sample {idx+1}")
            if ch_idx == len(channel_names) - 1:
                plt.xlabel("Samples")
            plt.ylabel("µV")
            if ch_idx == 0:
                plt.legend(loc='upper right')
            plt.tight_layout()
        figures.append(plt.gcf())
    return figures