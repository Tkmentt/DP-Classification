import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from collections import deque
from utils.utils import convert_to_microvolts
from utils import config as cfg
from tensorflow.keras.models import load_model
from preprocessing.prep_core import preprocess_eeg_for_cnn, is_artifact_free
from utils.utils import ensure_dir
import time
import datetime
import csv

# === Load CNN model ===
print("🔍 Loading raw CNN model...")
model_path = os.path.join(cfg.MODEL_DIR, "model_raw.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)
print(f"✅ Loaded CNN model from {model_path}")

# === Connect to EEG LSL Stream ===
print("🔍 Resolving EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=300)
if not streams:
    raise RuntimeError("No EEG stream found.")
inlet = StreamInlet(streams[0])
print("✅ Connected to EEG stream.")

# === Setup buffer ===
buffer_samples = cfg.WINDOW_SIZE  # 2000 samples (2 seconds)
step_samples = int(cfg.TESTING_STEP_SIZE * cfg.FS)  # e.g., 0.3 * 1000 = 300 samples

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
log_file = os.path.join(cfg.TESTING_LOG_DIR, f"predictions_{timestamp_now}.csv")
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

        # Only start predicting when we have at least 2s of data
        if len(eeg_buffer) == buffer_samples:
            batch_data = np.array(eeg_buffer)  # shape: (2000, 7)
            window_raw = batch_data[:, channel_indices]  # shape: (2000, 3)

            # === Convert to microvolts ===
            eeg_uV = convert_to_microvolts(window_raw)

            # === Preprocess for CNN ===
            eeg_preprocessed = preprocess_eeg_for_cnn(eeg_uV, sfreq=cfg.FS)

            # === Artifact rejection ===
            if is_artifact_free(eeg_preprocessed):
                # Reshape for CNN (batch_size, timesteps, channels)
                input_cnn = np.expand_dims(eeg_preprocessed, axis=0)

                prediction = model.predict(input_cnn, verbose=0)
                proba = prediction[0]

                proba_buffer.append(proba)

                # Keep only last 3 predictions
                if len(proba_buffer) > cfg.SMOOTHING_WINDOW:
                    proba_buffer.pop(0)

                if len(proba_buffer) == cfg.SMOOTHING_WINDOW:
                    # Average last 3 probability vectors
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
