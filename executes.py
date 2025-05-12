import os
import numpy as np
from datasets.data_loader import DataLoader
from typing import Tuple

pathX = "saved/X.npy"
pathy = "saved/y.npy"


def load_datasets(data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    print("=================== Load Datasets ===================")
    if os.path.exists(pathX) and os.path.exists(pathy):
        X, y = np.load(pathX), np.load(pathy)

        print("Loaded datasets ✅")
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        return X, y

    X, y = data_loader.load_data()
    np.save(pathX, X)
    np.save(pathy, y)

    print("Loaded datasets ✅")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y
