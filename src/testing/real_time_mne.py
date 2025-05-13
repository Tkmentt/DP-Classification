import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from collections import deque
from utils.utils import convert_to_microvolts, load_model
from preprocessing.prep_mne import convert_to_raw
from preprocessing.prep_core import is_artifact_free_mne, extract_features
from utils import config as cfg
from utils.utils import ensure_dir
import time
from joblib import load
import datetime
import csv

# Load CSP filters along with model
model, scaler = load_model(cfg.MODEL_DIR)
csp_path = os.path.join(cfg.MODEL_DIR, "csp_filters.pkl")
csp_filters = load(csp_path) if os.path.exists(csp_path) else None
print(f"✅ CSP filters loaded: shape {csp_filters.shape}" if csp_filters is not None else "⚠️ No CSP filters found")

# === Connect to EEG LSL Stream ===
print("🔍 Resolving EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=300)
if not streams:
    raise RuntimeError("No EEG stream found.")
inlet = StreamInlet(streams[0])
print("✅ Connected to EEG stream.")

# === Setup buffer ===
buffer_samples = cfg.WINDOW_SIZE  # 2000 samples = 2s
step_samples = int(cfg.TESTING_STEP_SIZE * cfg.FS)
eeg_buffer = deque(maxlen=buffer_samples)

channels = ['Fp1', 'Fp2', 'Cz', 'C3', 'C4', 'REF', 'GND']
target_channels = ['Cz', 'C3', 'C4']
channel_indices = [channels.index(ch) for ch in target_channels]

print(f"Buffering {buffer_samples} samples (2s window at {cfg.FS} Hz)")

# === Smoothing Setup ===
proba_buffer = []  # Store last 3 probability vectors for smoothing

# === Logging Setup ===
ensure_dir(cfg.TESTING_LOG_DIR)
timestamp_now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
log_file = os.path.join(cfg.TESTING_LOG_DIR, f"predictions_mne_{timestamp_now}.csv")
log_fields = ['timestamp', 'predicted_class', 'proba_class_0', 'proba_class_1']

with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(log_fields)

# === Real-Time Loop ===
while True:
    try:
        sample, timestamp = inlet.pull_sample(timeout=0.5)
        if sample is None:
            print("No data received. Stream might be closed.")
            time.sleep(1.0)
            continue
        eeg_buffer.append(sample)

        if len(eeg_buffer) == buffer_samples:
            batch_data = np.array(eeg_buffer)  # shape: (2000, 7)
            window_raw = batch_data[:, channel_indices]  # shape: (2000, 3)
            eeg_uV = convert_to_microvolts(window_raw)

            # === Wrap into MNE Raw and apply filters ===
            raw = convert_to_raw(eeg_uV, sfreq=cfg.FS)

            raw.notch_filter(
                freqs=cfg.NOTCH_TRESHOLD,
                method='iir',
                iir_params=dict(order=2, ftype='butter'),
                verbose=False
            )

            raw.filter(
                l_freq=1.0,
                h_freq=cfg.HIGH_BAND_THRESHOLD,
                method='iir',
                iir_params=dict(order=4, ftype='butter'),
                verbose=False
            )

            eeg_filtered = raw.get_data().T  # shape: (2000, 3)

            if is_artifact_free_mne(eeg_filtered, verbose=True):
                features = extract_features(np.expand_dims(eeg_filtered, axis=0), sfreq=cfg.FS, csp_filters=csp_filters)
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                probas = model.predict_proba(features_scaled)[0]

                proba_buffer.append(probas)

                if len(proba_buffer) > cfg.SMOOTHING_WINDOW:
                    proba_buffer.pop(0)

                if len(proba_buffer) == cfg.SMOOTHING_WINDOW:
                    avg_proba = np.mean(proba_buffer, axis=0)
                    final_predicted_class = np.argmax(avg_proba)

                    # Format timestamp
                    timestamp_now = datetime.datetime.now().strftime("%y.%m.%d-%H:%M:%S.%f")[:-3]

                    print(f"\n🧠 Smoothed Prediction: {final_predicted_class} | Averaged Probabilities: {avg_proba.round(3)}\n")

                    # Save to CSV
                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp_now,
                            int(final_predicted_class),
                            round(avg_proba[0], 6),
                            round(avg_proba[1], 6)
                        ])
     
            else:
                print("⚠️ Artifact detected. Skipping window.")


            # Slide window forward by step_size (300 samples)
            for _ in range(step_samples):
                if eeg_buffer:
                    eeg_buffer.popleft()

            time.sleep(cfg.TESTING_STEP_SIZE)  # 0.3 seconds

    except Exception as e:
        print(f"⚠️ LSL stream interrupted: {e}")
        time.sleep(1.0)
