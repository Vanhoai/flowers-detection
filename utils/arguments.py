import argparse
from enum import Enum


frameworks = ["keras", "pytorch"]
modes = ["train", "finetune", "plot", "load"]


class Modes(Enum):
    TRAIN = "train"
    FINETUNE = "finetune"
    PLOT = "plot"
    LOAD = "load"


def parse_arguments():
    description = "Flower Classification and Object Detection"
    parser = argparse.ArgumentParser(description=description)

    # Add arguments
    parser.add_argument(
        "--framework",
        type=str,
        default=frameworks[0],
        choices=frameworks,
        help="Framework to use (keras or pytorch)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=modes[0],
        choices=modes,
        help="Training mode (train from scratch or finetune)",
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=5,
        help="Number of layers to unfreeze in finetune mode",
    )

    parser.add_argument(
        "--images",
        type=int,
        default=10,
        help="Number of images to plot",
    )

    # Parse arguments
    args = parser.parse_args()

    print("=================== Parse Arguments ===================")
    print(f"Framework: {args.framework}")
    print(f"Mode: {args.mode}")
    print(f"Number of layers to unfreeze: {args.layers}")
    print(f"Number of images to plot: {args.images}")
    print("======================================================")

    return args
