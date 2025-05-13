import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import config as cfg


# === Function to parse ground truth intervals from a log file ===
def parse_ground_truth(file_path):
    intervals = []
    current_phase = None
    current_start = None

    # Read the file line by line
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue  # Skip empty lines

            # Parse timestamp and marker
            time_str, marker = line.strip().split(" ", 1)
            timestamp = datetime.datetime.strptime(time_str, "%y.%m.%d-%H:%M:%S.%f")

            # If marker is a start signal (S), record the start of the phase
            if marker.startswith("S"):
                current_phase = marker
                current_start = timestamp

            # If marker is an end signal (R), save the interval
            elif marker.startswith("R") and current_phase:
                label = cfg.EVENT_MAPPING.get(current_phase, None)
                if label is not None:
                    intervals.append({
                        "start": current_start,
                        "end": timestamp,
                        "label": label,
                        "phase": current_phase
                    })
                current_phase = None
                current_start = None

    return intervals

# === Function to load predictions from CSV ===
def load_predictions(file_path):
    df = pd.read_csv(file_path)
    # Convert timestamp string to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%y.%m.%d-%H:%M:%S.%f")
    return df

# === Function to match prediction timestamps to ground truth intervals ===
def match_labels(predictions, intervals):
    y_true, y_pred = [], []
    unmatched = 0  # Counter for unmatched prediction timestamps

    for _, row in predictions.iterrows():
        ts = row['timestamp']
        matched = False

        # Check if timestamp falls into any interval
        for interval in intervals:
            if interval['start'] <= ts <= interval['end']:
                y_true.append(interval['label'])
                y_pred.append(int(row['predicted_class']))
                matched = True
                break

        if not matched:
            unmatched += 1

    return y_true, y_pred, unmatched

# === Function to evaluate the performance of predictions ===
def evaluate(y_true, y_pred, output_dir, title, matched, unmatched):
    cm = confusion_matrix(y_true, y_pred)

    # Handle both binary and edge-case confusion matrix sizes
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # Calculate evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Prepare row for CSV log
    result_row = f"{title},{acc*100:.2f},{prec:.3f},{rec:.3f},{f1:.3f},{tp},{tn},{fp},{fn},{matched},{unmatched}\n"
    csv_path = os.path.join(output_dir, "results_summary.csv")
    header = "Trial,Accuracy,Precision,Recall,F1,TP,TN,FP,FN,Matched,Unmatched\n"

    # Write CSV file (append if it exists)
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(header)
    with open(csv_path, "a") as f:
        f.write(result_row)

    # Display metrics in terminal
    print(f"\n📊 {title} — Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

    # Plot and save confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Rest", "Move"], yticklabels=["Rest", "Move"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{title}.png"))
    plt.close()

# === Function to process all trials and modalities in a directory ===
def process_all_logs(base_dir="logs"):
    for trial_folder in os.listdir(base_dir):
        trial_path = os.path.join(base_dir, trial_folder)

        if not os.path.isdir(trial_path):
            continue

        # Process each modality separately
        for modality in ["features", "raw"]:
            sub_path = os.path.join(trial_path, modality)
            true_file = os.path.join(sub_path, "true.txt")
            pred_file = os.path.join(sub_path, "predictions.csv")

            if os.path.exists(true_file) and os.path.exists(pred_file):
                print(f"\n📂 Processing: {trial_folder}/{modality}")
                try:
                    # Parse and evaluate predictions
                    intervals = parse_ground_truth(true_file)
                    predictions = load_predictions(pred_file)
                    y_true, y_pred, unmatched = match_labels(predictions, intervals)
                    matched = len(y_true)
                    title = f"{trial_folder}_{modality}"
                    evaluate(y_true, y_pred, sub_path, title, matched, unmatched)
                except Exception as e:
                    print(f"❌ Error in {trial_folder}/{modality}: {e}")
            else:
                print(f"⚠️ Missing files in {trial_folder}/{modality}")

# === Entry point ===
if __name__ == "__main__":
    process_all_logs()