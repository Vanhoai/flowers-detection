import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import tensorflow as tf


def plot_sample_images(X, y, classes, n_samples=5, figsize=(15, 10)):
    """Hiển thị một số mẫu hình ảnh từ mỗi lớp"""
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, n_samples, figsize=figsize)

    for i in range(num_classes):
        # Lấy chỉ mục của các hình ảnh thuộc lớp i
        indices = np.where(np.argmax(y, axis=1) == i)[0]

        # Chọn ngẫu nhiên n_samples từ lớp
        selected_indices = np.random.choice(
            indices, size=min(n_samples, len(indices)), replace=False
        )

        for j, idx in enumerate(selected_indices):
            if num_classes > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]

            # Hiển thị hình ảnh
            img = X[idx]
            # Chuyển đổi từ [0,1] về [0,255] nếu cần thiết
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{classes[i]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, figsize=(10, 8)):
    """Vẽ ma trận nhầm lẫn"""
    y_true_classes = np.argmax(y_true, axis=1)
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


def visualize_model_predictions(model, X_sample, y_sample, classes, n_samples=5):
    """Hiển thị dự đoán của mô hình trên một số mẫu"""
    # Chọn ngẫu nhiên các mẫu
    indices = np.random.choice(range(len(X_sample)), n_samples, replace=False)
    X_display = X_sample[indices]
    y_true = np.argmax(y_sample[indices], axis=1)

    # Dự đoán
    predictions = model.predict(X_display)
    y_pred = np.argmax(predictions, axis=1)

    # Hiển thị
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))

    for i, (img, true_idx, pred_idx) in enumerate(zip(X_display, y_true, y_pred)):
        ax = axes[i]

        # Chuyển đổi từ [0,1] về [0,255] nếu cần thiết
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Đặt màu cho tiêu đề dựa trên dự đoán đúng/sai
        title_color = "green" if true_idx == pred_idx else "red"
        ax.set_title(
            f"True: {classes[true_idx]}\nPred: {classes[pred_idx]}", color=title_color
        )
        ax.axis("off")

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
