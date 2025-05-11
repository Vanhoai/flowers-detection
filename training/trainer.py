import os
import yaml
import tensorflow as tf
from keras import callbacks
import matplotlib.pyplot as plt
import numpy as np


class ModelTrainer:
    def __init__(self, model, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.model = model
        self.epochs = self.config["training"]["epochs"]
        self.checkpoint_path = self.config["training"]["checkpoint_path"]
        self.log_dir = self.config["training"]["tensorboard_log_dir"]

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_callbacks(self):
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=self.config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        )

        tensorboard = callbacks.TensorBoard(
            log_dir=self.log_dir, histogram_freq=1, write_graph=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
        )

        return [checkpoint_callback, early_stopping, tensorboard, reduce_lr]

    def train(self, X_train, y_train, X_val=None, y_val=None, validation_split=None):
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = (
                self.config["data"]["validation_split"]
                if validation_split is None
                else validation_split
            )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.config["data"]["batch_size"],
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=self.get_callbacks(),
            verbose=1,
        )

        return history

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="lower right")

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        plt.tight_layout()
        plt.show()

    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
        return results
