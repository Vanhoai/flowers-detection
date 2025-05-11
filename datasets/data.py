import os
import cv2
import matplotlib.pyplot as plt
from keras import preprocessing, legacy, callbacks
from keras import Sequential, layers, models
import numpy as np

classes = [
    "Hoàng Hậu",
    "Trang",
    "Trang Trắng",
    "Dừa Cạn",
    "Cúc",
    "Giấy",
    "Đại Tướng Quân",
    "Lan",
    "Loa Kèn",
    "Bằng Lăng",
]
datasets = "/Users/aurorastudyvn/Workspace/ML/flowers_detection/data/datasets"
flowers = "/Users/aurorastudyvn/Workspace/ML/flowers_detection/flowers"

IMAGE_LEN = 100
IMAGE_SIZE = (200, 200)
nums_classes = 10

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.4),
        layers.RandomZoom(0.2),
        # layers.RandomTranslation(0.2, 0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ]
)


def load_data():
    X = np.zeros((nums_classes * IMAGE_LEN, *IMAGE_SIZE, 3), dtype=int)
    y = np.zeros(nums_classes * IMAGE_LEN, dtype=int)
    for i in range(nums_classes):
        y[i * IMAGE_LEN : (i + 1) * IMAGE_LEN] = i

    for i in range(nums_classes):
        category = os.path.join(flowers, str(i + 1))
        files = os.listdir(category)

        XI = np.zeros((IMAGE_LEN, *IMAGE_SIZE, 3))
        count = 0

        size_images = files.__len__()
        size_gen = np.ceil(IMAGE_LEN / size_images).astype(int)
        for path in files:
            img = cv2.imread(os.path.join(category, path))
            if (img is None) or (count >= IMAGE_LEN):  # avoid .DS_Store file
                continue
            img = cv2.resize(img, (200, 200))

            XI[count] = img
            count += 1

            for _ in range(size_gen - 1):
                if count >= IMAGE_LEN:
                    break

                new_image = data_augmentation(img)
                XI[count] = new_image
                count += 1

        print(f"category: {category} -> {classes[i]}")
        print(f"Range: {i * IMAGE_LEN}: {(i + 1) * IMAGE_LEN}")
        X[i * IMAGE_LEN : (i + 1) * IMAGE_LEN] = XI

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

    return X_cleaned, y_cleaned


def split_dataset(X, y, test_split=0.2, shuffle=True):
    nums_train = int(X.shape[0] * (1 - test_split))

    if shuffle is True:
        indices = np.array(range(X.shape[0]))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    X_test, y_test = X[nums_train:], y[nums_train:]
    X_train, y_train = X[:nums_train], y[:nums_train]

    return X_train, y_train, X_test, y_test
