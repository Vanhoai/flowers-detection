from abc import ABC, abstractmethod
import tensorflow as tf
from keras import layers, models


class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None

    @abstractmethod
    def build(self):
        pass

    def compile(self, optimizer=None, loss=None, metrics=None):
        if not self.model:
            self.build()

        if not optimizer:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config["model"]["learning_rate"]
            )

        if not loss:
            loss = "categorical_crossentropy"

        if not metrics:
            metrics = ["accuracy"]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        if not self.model:
            self.build()

        return self.model.summary()

    def save(self, filepath):
        if not self.model:
            raise ValueError("Model hasn't been built yet.")
        self.model.save(filepath)

    def load(self, filepath):
        self.model = models.load_model(filepath)
        return self.model
