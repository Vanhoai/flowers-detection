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
    """Đánh giá các chỉ số hiệu suất cho bài toán phân loại"""
    # Chuyển đổi one-hot encoding thành các nhãn lớp
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

    # Tính toán các chỉ số
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(
        y_true_labels, y_pred_labels, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_true_labels, y_pred_labels, average="weighted", zero_division=0
    )
    f1 = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)

    # In kết quả
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Báo cáo chi tiết
    if classes is not None:
        print("\nClassification Report:")
        print(
            classification_report(
                y_true_labels, y_pred_labels, target_names=classes, zero_division=0
            )
        )

    # Tính ROC AUC nếu có xác suất dự đoán
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
    """Vẽ đường cong học (learning curves) từ lịch sử training"""
    metrics = list(history.history.keys())
    train_metrics = [m for m in metrics if not m.startswith("val_")]

    n_rows = len(train_metrics)
    plt.figure(figsize=(12, 4 * n_rows))

    for i, metric in enumerate(train_metrics):
        plt.subplot(n_rows, 1, i + 1)
        plt.plot(history.history[metric], label=f"Training {metric}")
        val_metric = f"val_{metric}"
        if val_metric in metrics:
            plt.plot(history.history[val_metric], label=f"Validation {metric}")
        plt.title(f"Model {metric}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def per_class_accuracy(y_true, y_pred, classes):
    """Tính độ chính xác cho từng lớp"""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    results = {}

    for i, class_name in enumerate(classes):
        # Chỉ mục của các mẫu thuộc lớp i
        class_indices = np.where(y_true == i)[0]

        if len(class_indices) > 0:
            # Độ chính xác trên các mẫu của lớp i
            class_accuracy = accuracy_score(
                y_true[class_indices], y_pred[class_indices]
            )
            results[class_name] = class_accuracy
            print(f"Accuracy for {class_name}: {class_accuracy:.4f}")
        else:
            results[class_name] = None
            print(f"No samples for class {class_name}")

    return results


def find_misclassified_samples(X, y_true, y_pred, classes, max_samples=10):
    """Tìm và trả về các mẫu bị phân loại sai"""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Tìm các chỉ mục mà dự đoán khác với nhãn thật
    misclassified_indices = np.where(y_true != y_pred)[0]

    # Lấy tối đa max_samples mẫu
    sample_indices = misclassified_indices[
        : min(max_samples, len(misclassified_indices))
    ]

    if len(sample_indices) == 0:
        print("No misclassified samples found!")
        return None

    # Tạo danh sách các mẫu bị phân loại sai
    misclassified_samples = {
        "indices": sample_indices,
        "X": X[sample_indices],
        "true_labels": y_true[sample_indices],
        "pred_labels": y_pred[sample_indices],
        "true_class_names": [classes[i] for i in y_true[sample_indices]],
        "pred_class_names": [classes[i] for i in y_pred[sample_indices]],
    }

    return misclassified_samples
