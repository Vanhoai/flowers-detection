import os
import numpy as np
from datasets.data_loader import DataLoader
from typing import Tuple

pathX = "saved/X.npy"
pathy = "saved/y.npy"

pathX_train = "saved/X_train.npy"
pathy_train = "saved/y_train.npy"

pathX_test = "saved/X_test.npy"
pathy_test = "saved/y_test.npy"


def load_datasets(
    data_loader: DataLoader, is_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if is_cache is True and os.path.exists(pathX) and os.path.exists(pathy):
        print("=================== Load Datasets From Cache ===================")
        X, y = np.load(pathX), np.load(pathy)
        print("Loaded datasets ✅")
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        return X, y

    print("=================== Load Datasets ===================")
    X, y = data_loader.load_data()
    np.save(pathX, X)
    np.save(pathy, y)

    print("Loaded datasets ✅")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


def split_datasets(data_loader: DataLoader, X: np.ndarray, y: np.ndarray):
    print("=================== Split Datasets ===================")
    X_train, y_train, X_test, y_test = data_loader.prepare_datasets(X, y)

    np.save(pathX_train, X_train)
    np.save(pathy_train, y_train)

    np.save(pathX_test, X_test)
    np.save(pathy_test, y_test)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Split datasets ✅")


def load_vars():
    X_train = np.load(pathX_train)
    y_train = np.load(pathy_train)

    X_test = np.load(pathX_test)
    y_test = np.load(pathy_test)

    return X_train, y_train, X_test, y_test
