from abc import ABC, abstractmethod
import tensorflow as tf
from keras import layers, models, optimizers
from enum import Enum
from typing import List


class Loss(Enum):
    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy"
    BINARY_CROSSENTROPY = "binary_crossentropy"


class BaseModel(ABC):
    def __init__(self, config):
        self.learning_rate = config["model"]["learning_rate"]
        self.model = None

    @abstractmethod
    def build(self): ...

    def compile(self, optimizer=None, loss: Loss = None, metrics: List[str] = None):
        if not self.model:
            self.model = self.build()

        if not optimizer:
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        if not loss:
            loss = "sparse_categorical_crossentropy"

        if not metrics:
            metrics = ["accuracy"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def predict(self, X_test):
        if not self.model:
            raise ValueError("Model hasn't been built yet.")

        return self.model.predict(X_test)

    def summary(self):
        if not self.model:
            raise ValueError("Model hasn't been built yet.")
        return self.model.summary()

    def load(self, filepath):
        self.model = models.load_model(filepath)
        return self.model
