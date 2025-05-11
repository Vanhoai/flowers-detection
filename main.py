import os
import yaml
import numpy as np
import tensorflow as tf
from data.data_loader import DataLoader
from models.flower_model import FlowerClassificationModel
from training.trainer import ModelTrainer
from utils.visualization import (
    plot_sample_images,
    plot_confusion_matrix,
    visualize_model_predictions,
)
from evaluation.metrics import (
    evaluate_model_classification,
    plot_learning_curves,
    per_class_accuracy,
)
import argparse


def main():
    # Đọc tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Flower Classification Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Đường dẫn đến file cấu hình",
    )
    parser.add_argument(
        "--is_new_training",
        action="store_true",
        help="Huấn luyện mô hình mới hoặc tiếp tục huấn luyện",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Fine tune mô hình sau khi huấn luyện ban đầu",
    )
    args = parser.parse_args()

    # Đọc cấu hình
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    print("======= Khởi tạo DataLoader =======")
    data_loader = DataLoader(args.config)

    print("======= Tải và xử lý dữ liệu =======")
    X, y = data_loader.load_data()

    print("======= Chia tập dữ liệu =======")
    X_train, y_train, X_val, y_val = data_loader.prepare_datasets(X, y)

    print("======= Hiển thị một số mẫu dữ liệu =======")
    plot_sample_images(X_train, y_train, config["classes"], n_samples=3)

    print("======= Xây dựng mô hình =======")
    model = FlowerClassificationModel(config)
    if args.is_new_training or not os.path.exists(
        config["training"]["checkpoint_path"]
    ):
        model.build()
    else:
        print(
            f"Tải mô hình đã được huấn luyện từ {config['training']['checkpoint_path']}"
        )
        model.load(config["training"]["checkpoint_path"])

    model.compile()
    model.summary()

    print("======= Huấn luyện mô hình =======")
    trainer = ModelTrainer(model.model, args.config)
    history = trainer.train(X_train, y_train, X_val, y_val)

    print("======= Vẽ đồ thị quá trình huấn luyện =======")
    trainer.plot_training_history(history)
    plot_learning_curves(history)

    if args.fine_tune:
        print("======= Fine-tuning mô hình =======")
        model.fine_tune()
        history_ft = trainer.train(X_train, y_train, X_val, y_val)
        print("======= Vẽ đồ thị quá trình fine-tuning =======")
        trainer.plot_training_history(history_ft)

    print("======= Đánh giá mô hình trên tập validation =======")
    y_pred = model.model.predict(X_val)
    evaluate_model_classification(y_val, y_pred, config["classes"])

    print("======= Ma trận nhầm lẫn =======")
    plot_confusion_matrix(y_val, y_pred, config["classes"])

    print("======= Độ chính xác cho từng lớp =======")
    per_class_accuracy(y_val, y_pred, config["classes"])

    print("======= Hiển thị một số dự đoán =======")
    visualize_model_predictions(model.model, X_val, y_val, config["classes"])

    print(f"Mô hình đã được lưu tại {config['training']['checkpoint_path']}")


if __name__ == "__main__":
    main()
