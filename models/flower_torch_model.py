import torch
import torch.nn as nn
import torchvision.models as models


class FlowerClassificationTorchModel:
    """
    Flexible PyTorch model for flower classification with support for multiple base models
    and easy fine-tuning.
    """

    # Mapping of supported base models
    SUPPORTED_MODELS = {
        "MobileNetV2": {
            "model_func": models.mobilenet_v2,
            "weights": models.MobileNet_V2_Weights.DEFAULT,
            "feature_dim": 1280,
        },
        "EfficientNetB0": {
            "model_func": models.efficientnet_b0,
            "weights": models.EfficientNet_B0_Weights.DEFAULT,
            "feature_dim": 1280,
        },
        "ResNet50": {
            "model_func": models.resnet50,
            "weights": models.ResNet50_Weights.DEFAULT,
            "feature_dim": 2048,
        },
        "DenseNet121": {
            "model_func": models.densenet121,
            "weights": models.DenseNet121_Weights.DEFAULT,
            "feature_dim": 1024,
        },
    }

    def __init__(self, config):
        self.config = config
        self.input_shape = tuple(config["model"]["input_shape"])
        self.num_classes = config["model"]["num_classes"]
        self.base_model_name = config["model"]["base_model_name"]
        self.dropout_rate = config.get("model", {}).get("dropout_rate", 0.5)

        # Validate base model
        if self.base_model_name not in self.SUPPORTED_MODELS:
            supported_models = ", ".join(self.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Unsupported base model: {self.base_model_name}. "
                f"Supported models are: {supported_models}"
            )

        self.model = None
        self.base_model = None

    def build(self):
        """
        Build the classification model with a flexible architecture

        :return: Constructed PyTorch model
        """
        # Get model configuration
        model_config = self.SUPPORTED_MODELS[self.base_model_name]
        model_func = model_config["model_func"]
        weights = model_config["weights"]
        feature_dim = model_config["feature_dim"]

        # Load pre-trained model
        self.base_model = model_func(weights=weights)

        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Create custom classifier
        classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )

        # Replace the original classifier
        if self.base_model_name.startswith("MobileNetV2"):
            self.base_model.classifier = classifier
            self.model = self.base_model
        elif self.base_model_name.startswith("EfficientNetB0"):
            self.base_model.classifier = classifier
            self.model = self.base_model
        elif self.base_model_name.startswith("ResNet50"):
            self.base_model.fc = classifier
            self.model = self.base_model
        elif self.base_model_name.startswith("DenseNet121"):
            self.base_model.classifier = classifier
            self.model = self.base_model

        return self.model

    def fine_tune(self, num_layers_to_unfreeze=5):
        """
        Fine-tune the model by unfreezing the last few layers of the base model

        :param num_layers_to_unfreeze: Number of layers to unfreeze from the end
        :return: Fine-tuned model
        """
        if self.model is None:
            raise ValueError("Model hasn't been built yet. Call build() first.")

        # Determine the base model's layers
        if self.base_model_name.startswith("MobileNetV2"):
            layers_to_unfreeze = self.base_model.features[-num_layers_to_unfreeze:]
        elif self.base_model_name.startswith("EfficientNetB0"):
            layers_to_unfreeze = self.base_model.features[-num_layers_to_unfreeze:]
        elif self.base_model_name.startswith("ResNet50"):
            layers_to_unfreeze = list(self.base_model.layer4.children())[
                -num_layers_to_unfreeze:
            ]
        elif self.base_model_name.startswith("DenseNet121"):
            layers_to_unfreeze = list(self.base_model.features.children())[
                -num_layers_to_unfreeze:
            ]
        else:
            raise ValueError(
                f"Unsupported model for fine-tuning: {self.base_model_name}"
            )

        # Unfreeze selected layers
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

        return self.model

    def get_optimizer(self, learning_rate=1e-4):
        """
        Create an optimizer for the model

        :param learning_rate: Learning rate for the optimizer
        :return: Configured optimizer
        """
        # Separate parameters with and without weight decay
        no_decay = ["bias", "BatchNorm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 1e-2,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate, weight_decay=1e-2
        )

    def get_loss_function(self):
        """
        Get the appropriate loss function for multi-class classification

        :return: Loss function
        """
        return nn.CrossEntropyLoss()

    @classmethod
    def add_custom_model(cls, model_name, model_func, weights, feature_dim):
        """
        Add a custom model to the supported models

        :param model_name: Name of the model
        :param model_func: Function to load the model
        :param weights: Pre-trained weights
        :param feature_dim: Dimension of the feature layer
        """
        cls.SUPPORTED_MODELS[model_name] = {
            "model_func": model_func,
            "weights": weights,
            "feature_dim": feature_dim,
        }
