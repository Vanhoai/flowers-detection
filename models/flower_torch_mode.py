from .base_model import BaseModel


class FlowerClassificationTorchModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = tuple(config["model"]["input_shape"])
        self.num_classes = config["model"]["num_classes"]
        self.base_model_name = config["model"]["base_model_name"]
        self.dropout_rate = config["model"]["dropout_rate"]

    def build(self):
        match self.base_model_name:
            case "VGG16":
                pass
            case "MobileNetV2":
                pass
            case "EfficientNetB0":
                pass
            case "ResNet50":
                pass
            case _:
                raise ValueError(f"Unsupported base model: {self.base_model_name}")

        # base_model.trainable = False
        # inputs = layers.Input(shape=self.input_shape)

        # # input layer
        # x = base_model(inputs, training=False)

        # x = layers.Flatten()(x)
        # x = layers.Dense(512, activation="relu")(x)
        # x = layers.Dense(256, activation="relu")(x)
        # x = layers.Dropout(self.dropout_rate)(x)

        # # output layer
        # outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        # self.model = models.Model(inputs, outputs)
        return self.model

    def fine_tune(self, num_layers_to_unfreeze=5):
        # if not self.model:
        #     raise ValueError("Model hasn't been built yet.")

        # base_model = None
        # for layer in self.model.layers:
        #     if hasattr(layer, "layers"):
        #         base_model = layer
        #         break

        # if not base_model:
        #     raise ValueError("Could not find the base model in the overall model.")

        # print("The number of layers in the base model:", len(base_model.layers))
        # print("The number of layers to unfreeze:", num_layers_to_unfreeze)
        # base_model.trainable = True
        # for layer in base_model.layers[:-num_layers_to_unfreeze]:
        #     layer.trainable = False

        return self.model
