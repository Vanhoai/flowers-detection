import os
import yaml
import argparse
import numpy as np
from datasets.data_loader import DataLoader
from utils.visualization import (
    plot_sample_images,
    plot_confusion_matrix,
    visualize_model_predictions,
)
from models.flower_keras_model import FlowerClassificationKerasModel
from training.trainer import ModelTrainer
from evaluation.metrics import (
    evaluate_model_classification,
    plot_learning_curves,
    per_class_accuracy,
)
from utils.arguments import parse_arguments, Modes
from executes import load_datasets


def main():
    args = parse_arguments()
    args.config = "config/config.yaml"

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    data_loader = DataLoader(args.config)
    match args.mode:
        case Modes.LOAD.value:
            X, y = load_datasets(data_loader)
            X_train, y_train, X_test, y_test = data_loader.prepare_datasets(X, y)
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            pass

        case Modes.TRAIN.value:
            print("Train")
            pass

        case Modes.FINETUNE.value:
            print("Finetune")
            pass

        case Modes.PLOT.value:
            print("Plot")
            pass

    # print("=================== Show Sample Images ===================")
    # plot_sample_images(X_train, y_train, config["classes"])

    # print("=================== Build Model ===================")
    # flower_classification_model = FlowerClassificationKerasModel(config)
    # if args.is_new_training or not os.path.exists(
    #     config["training"]["checkpoint_path"]
    # ):
    #     print("Build new model from scratch")
    #     flower_classification_model.build()
    # else:
    #     print(f"Load model from {config['training']['checkpoint_path']}")
    #     flower_classification_model.load(config["training"]["checkpoint_path"])

    # flower_classification_model.compile()
    # flower_classification_model.summary()

    # print("=================== Training Model ===================")
    # trainer = ModelTrainer(flower_classification_model.model, args.config)
    # hist = trainer.train(X_train, y_train)
    # np.save("saved/hist.npy", hist)

    # print("=================== Draw Learning Curves ===================")
    # items = np.load("saved/hist.npy", allow_pickle=True).item()
    # trainer.plot_training_history(items.history)
    # plot_learning_curves(items.history)

    # if args.fine_tune:
    #     print("=================== Fine-tuning Model ===================")
    #     flower_classification_model = flower_classification_model.fine_tune()
    #     flower_classification_model.compile()
    #     flower_classification_model.summary()
    #     history_ft = trainer.train(X_train, y_train)
    #     print("=================== Draw Learning Curves ===================")
    #     trainer.plot_training_history(history_ft.history)

    # print("=================== Prediction ===================")
    # y_pred = flower_classification_model.predict(X_test)
    # evaluate_model_classification(y_test, y_pred, config["classes"])

    # print("=================== Confusion Matrix ===================")
    # plot_confusion_matrix(y_test, y_pred, config["classes"])

    # print("=================== Accuracy per Class ===================")
    # per_class_accuracy(y_test, y_pred, config["classes"])

    # print("=================== Visualize Image Predicted ===================")
    # y_pred = flower_classification_model.predict(X_test)
    # visualize_model_predictions(X_test, y_pred, config["classes"])


if __name__ == "__main__":
    main()
