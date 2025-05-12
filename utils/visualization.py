import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import tensorflow as tf


def plot_sample_images(X, y, classes, n_samples=10):
    n_cols = 5
    n_rows = 2

    width = 4 * n_cols
    height = 4 * n_rows

    plt.close("all")
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    for i in range(n_samples):
        r = i // n_cols
        c = i % n_cols

        img = X[i]
        label = classes[y[i]]
        if n_rows > 1:
            axs[r, c].imshow(img)
            axs[r, c].set_title(label)
        else:
            axs[c].imshow(img)
            axs[c].set_title(label)

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, figsize=(10, 8)):
    y_true_classes = y_true
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def visualize_model_predictions(X, y, classes, n_samples=10):
    y = np.argmax(y, axis=1)

    n_cols = 5
    n_rows = 2

    width = 4 * n_cols
    height = 4 * n_rows

    plt.close("all")
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    for i in range(n_samples):
        r = i // n_cols
        c = i % n_cols

        img = X[i]
        label = classes[y[i]]
        if n_rows > 1:
            axs[r, c].imshow(img)
            axs[r, c].set_title(label)
        else:
            axs[c].imshow(img)
            axs[c].set_title(label)

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_activation_maps(model, img, layer_name, figsize=(15, 8)):
    """Hiển thị bản đồ kích hoạt của một lớp cho một hình ảnh"""
    # Tạo mô hình mới với đầu ra là lớp được chọn
    activation_model = tf.keras.models.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )

    # Mở rộng kích thước hình ảnh để phù hợp với đầu vào của mô hình
    img_array = np.expand_dims(img, axis=0)

    # Lấy kích hoạt
    activations = activation_model.predict(img_array)

    # Số lượng feature maps để hiển thị
    n_features = min(16, activations.shape[-1])

    # Kích thước của hình
    size = activations.shape[1]

    # Tạo lưới để hiển thị
    cols = 4
    rows = n_features // cols + (1 if n_features % cols > 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        if i < n_features:
            # Hiển thị feature map
            ax.imshow(activations[0, :, :, i], cmap="viridis")
            ax.set_title(f"Feature Map {i+1}")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
