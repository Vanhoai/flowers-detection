import os
import numpy as np
import cv2
import tensorflow as tf
import yaml
from keras.src.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
    RandomBrightness,
)


class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        print(self.config)
        # self.flowers_path = self.config["data"]["train_path"]
        # self.image_size = tuple(self.config["data"]["image_size"])
        # self.classes = self.config["classes"]
        # self.num_classes = len(self.classes)
        # self.image_per_class = self.config["data"]["image_per_class"]

        # # Tạo data augmentation pipeline
        # self.data_augmentation = tf.keras.Sequential(
        #     [
        #         (
        #             RandomFlip("horizontal_and_vertical")
        #             if self.config["augmentation"]["horizontal_flip"]
        #             and self.config["augmentation"]["vertical_flip"]
        #             else None
        #         ),
        #         RandomRotation(self.config["augmentation"]["rotation_range"]),
        #         RandomZoom(self.config["augmentation"]["zoom_range"]),
        #         RandomContrast(self.config["augmentation"]["contrast_range"]),
        #         RandomBrightness(self.config["augmentation"]["brightness_range"]),
        #     ]
        # )

        # # Loại bỏ các layer None
        # self.data_augmentation.layers = [
        #     layer for layer in self.data_augmentation.layers if layer is not None
        # ]

    # def load_data(self):
    #     X = np.zeros(
    #         (self.num_classes * self.image_per_class, *self.image_size, 3), dtype=int
    #     )
    #     y = np.zeros(self.num_classes * self.image_per_class, dtype=int)

    #     for i in range(self.num_classes):
    #         y[i * self.image_per_class : (i + 1) * self.image_per_class] = i

    #     # Tải và tăng cường dữ liệu
    #     for i in range(self.num_classes):
    #         category = os.path.join(self.flowers_path, str(i + 1))
    #         files = os.listdir(category)

    #         XI = np.zeros((self.image_per_class, *self.image_size, 3))
    #         count = 0

    #         size_images = len(files)
    #         size_gen = np.ceil(self.image_per_class / size_images).astype(int)

    #         for path in files:
    #             img_path = os.path.join(category, path)
    #             if not os.path.isfile(img_path) or path.startswith("."):
    #                 continue

    #             img = cv2.imread(img_path)
    #             if img is None or count >= self.image_per_class:
    #                 continue

    #             img = cv2.resize(img, self.image_size)

    #             XI[count] = img
    #             count += 1

    #             # Generate augmented images
    #             for _ in range(size_gen - 1):
    #                 if count >= self.image_per_class:
    #                     break

    #                 img_tensor = tf.convert_to_tensor(img)
    #                 new_image = self.data_augmentation(tf.expand_dims(img_tensor, 0))[0]
    #                 XI[count] = new_image.numpy()
    #                 count += 1

    #         print(f"Category: {category} -> {self.classes[i]}")
    #         print(
    #             f"Range: {i * self.image_per_class}: {(i + 1) * self.image_per_class}"
    #         )
    #         X[i * self.image_per_class : (i + 1) * self.image_per_class] = XI

    #     # Loại bỏ ảnh lỗi (ma trận 0)
    #     non_zero_indices = []
    #     for i in range(X.shape[0]):
    #         if np.sum(X[i]) > 0:
    #             non_zero_indices.append(i)

    #     X_cleaned = X[non_zero_indices]
    #     y_cleaned = y[non_zero_indices]

    #     print(f"Original dataset size: {X.shape[0]}")
    #     print(f"Cleaned dataset size: {X_cleaned.shape[0]}")
    #     print(f"Removed {X.shape[0] - X_cleaned.shape[0]} zero matrices")

    #     # Chuẩn hóa dữ liệu
    #     X_cleaned = X_cleaned.astype("float32") / 255.0

    #     return X_cleaned, y_cleaned

    # def prepare_datasets(self, X, y, test_size=None):
    #     """Chia dữ liệu thành tập train và validation"""
    #     if test_size is None:
    #         test_size = self.config["data"]["validation_split"]

    #     # Shuffle data
    #     indices = np.arange(X.shape[0])
    #     np.random.shuffle(indices)
    #     X = X[indices]
    #     y = y[indices]

    #     # Chia tập dữ liệu
    #     split_idx = int(X.shape[0] * (1 - test_size))
    #     X_train, X_val = X[:split_idx], X[split_idx:]
    #     y_train, y_val = y[:split_idx], y[split_idx:]

    #     # One-hot encoding cho nhãn
    #     y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    #     y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)

    #     return X_train, y_train, X_val, y_val
