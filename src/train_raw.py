import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from preprocessing.prep_raw import load_full_dataset
from classification.keras.CNN import build_cnn_model, get_lr_scheduler, get_early_stopping
from utils.utils_plot import plot_confusion_matrix, plot_cv_accuracy, plot_cv_loss, plot_aggregated_confusion_matrix
from utils.utils import save_figure, save_raw_model, save_classification_report, get_plot_theme
from utils import config as cfg
import os


def cross_validate_and_retrain_best_model(windows, labels, group_ids, output_dir=cfg.OUTPUT_RAW_DIR,
                                          n_splits=cfg.KFOLD_SPLITS, use_group_kfold=True, random_state=42):
    # Detect fold type and get visual theme
    folding_type = "GroupKFold" if use_group_kfold and group_ids is not None else "StratifiedKFold"
    theme = get_plot_theme(data_type="RAW", folding_type=folding_type)

    print(f"🔀 Using {folding_type} with RAW theme")
    kf = GroupKFold(n_splits=n_splits) if folding_type == "GroupKFold" else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = kf.split(windows, labels, group_ids) if use_group_kfold and group_ids is not None else kf.split(windows, labels)

    acc_scores = []
    train_losses = []
    val_losses = []
    best_acc = -1
    best_model_weights = None

    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n🔍 Fold {fold_idx + 1}/{n_splits}")

        X_train_raw = windows[train_idx]
        X_test_raw = windows[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        y_train_cat = np.stack([1 - y_train, y_train], axis=1)
        y_test_cat = np.stack([1 - y_test, y_test], axis=1)

        cnn_model = build_cnn_model(input_shape=(X_train_raw.shape[1], X_train_raw.shape[2]))
        history = cnn_model.fit(
            X_train_raw, y_train_cat,
            validation_data=(X_test_raw, y_test_cat),
            epochs=80,
            batch_size=32,
            callbacks=[get_lr_scheduler(), get_early_stopping()],
            verbose=1
        )

        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        cnn_probs = cnn_model.predict(X_test_raw)
        cnn_preds = np.argmax(cnn_probs, axis=1)

        hard_mask = (y_test == 0) | (y_test == 1)
        y_test_hard = y_test[hard_mask].astype(int)
        cnn_preds_hard = cnn_preds[hard_mask]

        acc = accuracy_score(y_test_hard, cnn_preds_hard)
        acc_scores.append(acc)

        print("🧠 CNN Validation Results (Hard Labels Only):")
        report = classification_report(y_test_hard, cnn_preds_hard, digits=3)
        print(report)
        save_classification_report(report, fold=fold_idx + 1, output_dir=output_dir, prefix=theme["prefix"])

        cm = plot_confusion_matrix(
            y_test_hard, cnn_preds_hard, class_labels=[0, 1],
            title=f"{theme['prefix'].upper()} Fold {fold_idx + 1} Confusion Matrix",
            cmap=theme["cmap"]
        )
        save_figure(cm, os.path.join(output_dir, f"{theme['prefix']}_confusion_matrix_fold{fold_idx + 1}.png"))

        all_y_true.append(y_test_hard)
        all_y_pred.append(cnn_preds_hard)

        if acc > best_acc:
            best_acc = acc
            best_model_weights = cnn_model.get_weights()
            print(f"💾 New best model found in Fold {fold_idx + 1} with accuracy {acc:.3f}")

    # === Aggregated confusion matrices ===
    fig_cm, _ = plot_aggregated_confusion_matrix(
        y_trues=all_y_true,
        y_preds=all_y_pred,
        class_labels=[0, 1],
        normalize=False,
        title=f"{theme['prefix'].upper()} Final Aggregated Confusion Matrix",
        cmap=theme["cmap"]
    )
    save_figure(fig_cm, os.path.join(output_dir, f"{theme['prefix']}_confusion_matrix_final_aggregated.png"))

    fig_cm_norm, _ = plot_aggregated_confusion_matrix(
        y_trues=all_y_true,
        y_preds=all_y_pred,
        class_labels=[0, 1],
        normalize=True,
        title=f"{theme['prefix'].upper()} Final Aggregated Confusion Matrix (Normalized)",
        cmap=theme["cmap"]
    )
    save_figure(fig_cm_norm, os.path.join(output_dir, f"{theme['prefix']}_confusion_matrix_final_normalized.png"))

    # === Plot and save accuracy and loss curves ===
    acc_plot = plot_cv_accuracy(acc_scores, title=f"{theme['prefix'].upper()} Cross-Validation Accuracy per Fold", color=theme["line_color"])
    save_figure(acc_plot, os.path.join(output_dir, f"{theme['prefix']}_cv_accuracy_plot.png"))

    loss_plot = plot_cv_loss(train_losses, val_losses, title=f"{theme['prefix'].upper()} Cross-Validation Loss per Fold", color=theme["line_color"])
    save_figure(loss_plot, os.path.join(output_dir, f"{theme['prefix']}_cv_loss_plot.png"))

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

    use_group_kfold = True  # or False for StratifiedKFold

    cross_validate_and_retrain_best_model(windows, labels, group_ids, use_group_kfold=use_group_kfold)
