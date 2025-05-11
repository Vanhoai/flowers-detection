import tensorflow as tf
from keras import layers, models
from keras.src.applications import mobilenet_v2, efficientnet, resnet
from .base_model import BaseModel


class FlowerClassificationModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = tuple(config["model"]["input_shape"])
        self.num_classes = config["model"]["num_classes"]
        self.base_model_name = config["model"]["base_model"]
        self.dropout_rate = config["model"]["dropout_rate"]

    def build(self):
        if self.base_model_name == "MobileNetV2":
            base_model = mobilenet_v2.MobileNetV2(
                input_shape=self.input_shape, include_top=False, weights="imagenet"
            )
        elif self.base_model_name == "EfficientNetB0":
            base_model = efficientnet.EfficientNetB0(
                input_shape=self.input_shape, include_top=False, weights="imagenet"
            )
        elif self.base_model_name == "ResNet50":
            base_model = resnet.ResNet50(
                input_shape=self.input_shape, include_top=False, weights="imagenet"
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")

        base_model.trainable = False

        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)
        return self.model

    def fine_tune(self, num_layers_to_unfreeze=10):
        if not self.model:
            raise ValueError("Model hasn't been built yet.")

        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                base_model = layer
                break

        if not base_model:
            raise ValueError("Could not find the base model in the overall model.")

        base_model.trainable = True
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config["model"]["learning_rate"] / 10
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.model
