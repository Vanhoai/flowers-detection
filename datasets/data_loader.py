import os
import numpy as np
import cv2
import yaml
import scipy.io as sio
from typing import Tuple
import torchvision
from keras import Sequential, utils
from keras.src.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
    RandomBrightness,
    RandomTranslation,
)

pathX = "saved/X.npy"
pathy = "saved/y.npy"

pathX_train = "saved/X_train.npy"
pathy_train = "saved/y_train.npy"

pathX_test = "saved/X_test.npy"
pathy_test = "saved/y_test.npy"


class DataLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # data
        self.datasets_path = self.config["data"]["datasets_path"]

        # classes
        self.classes = self.config["classes"]
        self.num_classes = len(self.classes)

        self.image_size = tuple(self.config["data"]["image_size"])
        self.batch_size = int(self.config["data"]["batch_size"])
        self.test_split = float(self.config["data"]["test_split"])
        self.validation_split = float(self.config["data"]["validation_split"])
        self.ipc = int(self.config["data"]["image_per_class"])
        self.is_cache = self.config["data"]["is_cache"]

        # create data augmentation pipeline
        self.data_augmentation = Sequential(
            [
                RandomFlip("horizontal_and_vertical"),
                RandomRotation(self.config["augmentation"]["rotation_range"]),
                RandomZoom(self.config["augmentation"]["zoom_range"]),
                RandomContrast(self.config["augmentation"]["contrast_range"]),
                RandomBrightness(self.config["augmentation"]["brightness_range"]),
            ]
        )

        has_translation_range = (
            "translation_range" in self.config["augmentation"].keys()
        )
        if has_translation_range:
            random_translation = RandomTranslation(
                self.config["augmentation"]["translation_range"][0],
                self.config["augmentation"]["translation_range"][1],
            )
            self.data_augmentation.add(random_translation)

    def load_data(self):
        if self.is_cache is True and os.path.exists(pathX) and os.path.exists(pathy):
            print("=================== Load Datasets From Cache ===================")
            X, y = np.load(pathX), np.load(pathy)
            return X, y

        print("=================== Load Datasets ===================")
        X = np.zeros((self.num_classes * self.ipc, *self.image_size, 3), dtype=int)
        y = np.zeros(self.num_classes * self.ipc, dtype=int)

        for i in range(self.num_classes):
            y[i * self.ipc : (i + 1) * self.ipc] = i

        for i in range(self.num_classes):
            category = os.path.join(self.datasets_path, str(i + 1))
            files = os.listdir(category)

            XI = np.zeros((self.ipc, *self.image_size, 3))
            count = 0

            size_images = files.__len__()
            size_gen = np.ceil(self.ipc / size_images).astype(int)
            for path in files:
                img = cv2.imread(os.path.join(category, path))
                if (img is None) or count >= self.ipc:  # avoid .DS_Store file
                    continue
                img = cv2.resize(img, (200, 200))

                XI[count] = img
                count += 1

                for _ in range(size_gen - 1):
                    if count >= self.ipc:
                        break

                    new_image = self.data_augmentation(img)
                    XI[count] = new_image
                    count += 1

            print(f"category: {category} -> {self.classes[i]}")
            print(f"Range: {i * self.ipc}: {(i + 1) * self.ipc}")
            X[i * self.ipc : (i + 1) * self.ipc] = XI

        # ignore image gen error -> zero matrix
        non_zero_indices = []
        for i in range(X.shape[0]):
            # Check if the image is not a zero matrix
            if np.sum(X[i]) > 0:
                non_zero_indices.append(i)

        # Keep only non-zero images and their corresponding labels
        X_cleaned = X[non_zero_indices]
        y_cleaned = y[non_zero_indices]

        # Print how many images were removed
        print(f"Original dataset size: {X.shape[0]}")
        print(f"Cleaned dataset size: {X_cleaned.shape[0]}")
        print(f"Removed {X.shape[0] - X_cleaned.shape[0]} zero matrices")

        # Replace original X and y with cleaned versions
        X = X_cleaned
        y = y_cleaned

        np.save(pathX, X)
        np.save(pathy, y)

        return X, y

    def prepare_datasets(
        self,
        X: np.array,
        y: np.array,
        test_split=None,
        is_shuffle=True,
        is_onehot=False,
        is_use_cache=True,
        is_save_cache=True,
    ):
        if (
            is_use_cache is True
            and os.path.exists(pathX_train)
            and os.path.exists(pathX_test)
            and os.path.exists(pathy_train)
            and os.path.exists(pathy_test)
        ):
            X_train, y_train = np.load(pathX_train), np.load(pathy_train)
            X_test, y_test = np.load(pathX_test), np.load(pathy_test)

            return X_train, y_train, X_test, y_test

        if test_split is None:
            test_split = self.test_split

        # Shuffle
        if is_shuffle is True:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

        # Split dataset
        split_idx = int(X.shape[0] * (1 - test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # One-hot encoding
        if is_onehot is True:
            y_train = utils.to_categorical(y_train, self.num_classes)
            y_test = utils.to_categorical(y_test, self.num_classes)

        if is_save_cache is True:
            np.save("saved/X_train.npy", X_train)
            np.save("saved/y_train.npy", y_train)
            np.save("saved/X_test.npy", X_test)
            np.save("saved/y_test.npy", y_test)

        return X_train, y_train, X_test, y_test
