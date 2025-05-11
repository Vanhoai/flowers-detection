import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
import matplotlib.pyplot as plt


def evaluate_model_classification(y_true, y_pred, classes=None):
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_pred_probs = y_pred
    else:
        y_pred_labels = y_pred
        y_pred_probs = None

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(
        y_true_labels, y_pred_labels, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_true_labels, y_pred_labels, average="weighted", zero_division=0
    )
    f1 = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if classes is not None:
        print("\nClassification Report:")
        print(
            classification_report(
                y_true_labels, y_pred_labels, target_names=classes, zero_division=0
            )
        )

    if y_pred_probs is not None and y_true.ndim > 1 and y_true.shape[1] > 1:
        try:
            auc = roc_auc_score(
                y_true, y_pred_probs, average="weighted", multi_class="ovr"
            )
            print(f"ROC AUC Score: {auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")

    results = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return results


def plot_learning_curves(history):
    metrics = list(history.keys())
    train_metrics = [m for m in metrics if not m.startswith("val_")]

    n_rows = 1
    n_cols = len(train_metrics)

    width = 4 * n_cols
    height = 4 * n_rows

    plt.close("all")
    _, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))

    for i, metric in enumerate(train_metrics):
        axs[i].plot(history[metric], label=f"Training {metric}")
        val_metric = f"val_{metric}"
        if val_metric in metrics:
            axs[i].plot(history[val_metric], label=f"Validation {metric}")

        axs[i].set_title(f"Model {metric}")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_xlabel("Epoch")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def per_class_accuracy(y_true, y_pred, classes):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    results = {}
    for i, class_name in enumerate(classes):
        class_indices = np.where(y_true == i)[0]

        if len(class_indices) > 0:
            class_accuracy = accuracy_score(
                y_true[class_indices], y_pred[class_indices]
            )
            results[class_name] = class_accuracy
            print(f"Accuracy for {class_name}: {class_accuracy:.4f}")
        else:
            results[class_name] = None
            print(f"No samples for class {class_name}")

    return results
