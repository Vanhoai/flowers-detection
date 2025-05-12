import tensorflow as tf
from keras import layers, models
from keras.src.applications import mobilenet_v2, efficientnet, resnet, vgg16
from .base_model import BaseModel


class FlowerClassificationKerasModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = tuple(config["model"]["input_shape"])
        self.num_classes = config["model"]["num_classes"]
        self.base_model_name = config["model"]["base_model_name"]
        self.dropout_rate = config["model"]["dropout_rate"]

    def build(self) -> models.Model:
        match self.base_model_name:
            case "VGG16":
                base_model = vgg16.VGG16(
                    input_shape=self.input_shape, include_top=False, weights="imagenet"
                )
            case "MobileNetV2":
                base_model = mobilenet_v2.MobileNetV2(
                    input_shape=self.input_shape, include_top=False, weights="imagenet"
                )
            case "EfficientNetB0":
                base_model = efficientnet.EfficientNetB0(
                    input_shape=self.input_shape, include_top=False, weights="imagenet"
                )
            case "ResNet50":
                base_model = resnet.ResNet50(
                    input_shape=self.input_shape, include_top=False, weights="imagenet"
                )
            case _:
                raise ValueError(f"Unsupported base model: {self.base_model_name}")

        base_model.trainable = False
        inputs = layers.Input(shape=self.input_shape)

        # input layer
        x = base_model(inputs, training=False)

        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # output layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)
        return self.model

    def fine_tune(self, num_layers_to_unfreeze=5):
        if not self.model:
            raise ValueError("Model hasn't been built yet.")

        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                base_model = layer
                break

        if not base_model:
            raise ValueError("Could not find the base model in the overall model.")

        print("The number of layers in the base model:", len(base_model.layers))
        print("The number of layers to unfreeze:", num_layers_to_unfreeze)
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

        return self.model
