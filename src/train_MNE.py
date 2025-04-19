import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import joblib

# === Load and stack all features ===
def load_all_preprocessed_features(data_folder="preprocessed_data"):
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

# === Train Classifier ===
def train_classifier(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=500,
              early_stopping=True, random_state=42)

    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_val_scaled)
    print("📊 Validation Report:")
    print(classification_report(y_val, y_pred, digits=3))
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    return clf, scaler

# === Main Script ===
if __name__ == "__main__":
    X, y, subjects = load_all_preprocessed_features()
    print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    clf, scaler = train_classifier(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/MLP_classifier.joblib")
    joblib.dump(scaler, "models/feature_scaler.joblib")
    print("💾 Model and scaler saved to 'models/' folder.")
