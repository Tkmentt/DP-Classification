import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score
from preprocessing.prep_raw import load_full_dataset
from classification.keras.CNN import build_cnn_model, get_lr_scheduler, get_early_stopping
from utils.utils_plot import plot_confusion_matrix
from utils.utils import save_figure, save_raw_model
from utils import config as cfg
import os


def cross_validate_and_retrain_best_model(windows, labels, group_ids, output_dir=cfg.OUTPUT_RAW_DIR, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(windows, labels, groups=group_ids))
    print(f"\n🔁 Prepared {n_splits}-fold GroupKFold splits.")

    acc_scores = []
    best_acc = -1
    best_model_weights = None

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n🔍 Fold {fold_idx + 1}/{n_splits}")

        X_train_raw = windows[train_idx]
        X_test_raw = windows[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        y_train_cat = np.stack([1 - y_train, y_train], axis=1)
        y_test_cat = np.stack([1 - y_test, y_test], axis=1)

        cnn_model = build_cnn_model(input_shape=(X_train_raw.shape[1], X_train_raw.shape[2]))
        cnn_model.fit(
            X_train_raw, y_train_cat,
            validation_data=(X_test_raw, y_test_cat),
            epochs=50,
            batch_size=32,
            callbacks=[get_lr_scheduler(), get_early_stopping()],
            verbose=1
        )

        cnn_probs = cnn_model.predict(X_test_raw)
        cnn_preds = np.argmax(cnn_probs, axis=1)

        hard_mask = (y_test == 0) | (y_test == 1)
        y_test_hard = y_test[hard_mask].astype(int)
        cnn_preds_hard = cnn_preds[hard_mask]

        acc = accuracy_score(y_test_hard, cnn_preds_hard)
        acc_scores.append(acc)

        print("🧠 CNN Validation Results (Hard Labels Only):")
        print(classification_report(y_test_hard, cnn_preds_hard, digits=3))

        cm = plot_confusion_matrix(y_test_hard, cnn_preds_hard, class_labels=[0, 1], title=f"Fold {fold_idx + 1} Confusion Matrix")
        save_figure(cm, os.path.join(output_dir, f"RAW_confusion_matrix_fold{fold_idx + 1}.png"))

        if acc > best_acc:
            best_acc = acc
            best_model_weights = cnn_model.get_weights()
            print(f"💾 New best model found in Fold {fold_idx + 1} with accuracy {acc:.3f}")

    # === Retrain best model on full dataset ===
    print("\n🚀 Retraining best model on full dataset...")
    y_cat_full = np.stack([1 - labels, labels], axis=1)
    best_model = build_cnn_model(input_shape=(windows.shape[1], windows.shape[2]))
    best_model.set_weights(best_model_weights)
    best_model.fit(
        windows, y_cat_full,
        epochs=15,
        batch_size=32,
        callbacks=[get_lr_scheduler(), get_early_stopping()],
        verbose=1
    )

    # === Save the final retrained model ===
    save_raw_model(best_model)


if __name__ == "__main__":
    print("🔍 Loading raw EEG dataset...")
    windows, labels, group_ids = load_full_dataset(cfg.DATA_DIR)
    print(f"✅ Loaded {windows.shape[0]} windows from {len(np.unique(group_ids))} subjects.")

    cross_validate_and_retrain_best_model(windows, labels, group_ids)
