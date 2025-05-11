import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def load_and_preprocess_image(image_path, target_size=(200, 200)):
    """Tải và tiền xử lý một hình ảnh"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Resize
    img = cv2.resize(img, target_size)

    # Chuẩn hóa
    img = img.astype("float32") / 255.0

    return img


def load_and_preprocess_images_from_directory(
    directory, classes, target_size=(200, 200), max_per_class=10
):
    """Tải và tiền xử lý các hình ảnh từ thư mục"""
    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, str(class_idx + 1))
        files = os.listdir(class_dir)

        count = 0
        for file in files:
            if not file.startswith(".") and count < max_per_class:
                try:
                    img_path = os.path.join(class_dir, file)
                    img = load_and_preprocess_image(img_path, target_size)
                    images.append(img)
                    labels.append(class_idx)
                    count += 1
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    return np.array(images), np.array(labels)


def apply_augmentation(image, data_augmentation=None):
    """Áp dụng augmentation cho một hình ảnh"""
    if data_augmentation is None:
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.2),
                tf.keras.layers.RandomBrightness(0.2),
            ]
        )

    # Expand dimension để khớp với đầu vào của data_augmentation
    img_tensor = tf.convert_to_tensor(image)
    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, 0)

    # Áp dụng augmentation
    augmented_img = data_augmentation(img_tensor)[0]

    return augmented_img.numpy()


def show_augmentation_examples(image, n_examples=5, figsize=(15, 3)):
    """Hiển thị nhiều phiên bản tăng cường của một hình ảnh"""
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ]
    )

    plt.figure(figsize=figsize)

    # Hiển thị hình ảnh gốc
    plt.subplot(1, n_examples + 1, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    # Hiển thị các phiên bản tăng cường
    for i in range(n_examples):
        augmented = apply_augmentation(image, data_augmentation)
        plt.subplot(1, n_examples + 1, i + 2)
        plt.imshow(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))
        plt.title(f"Augmented {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
