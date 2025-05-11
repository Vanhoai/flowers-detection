import os
import yaml
import numpy as np
import argparse
import cv2
import tensorflow as tf
from keras import models
from utils.image_processing import load_and_preprocess_image, show_augmentation_examples
import matplotlib.pyplot as plt


def predict_single_image(model, image_path, config):
    """Dự đoán lớp cho một hình ảnh"""
    img = load_and_preprocess_image(
        image_path, target_size=tuple(config["model"]["input_shape"][:2])
    )

    # Mở rộng kích thước để phù hợp với đầu vào của mô hình
    img_array = np.expand_dims(img, axis=0)

    # Dự đoán
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = config["classes"][predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    # Hiển thị kết quả
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    # Hiển thị top-3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    print("\nTop 3 predictions:")
    for idx in top_indices:
        print(f"{config['classes'][idx]}: {predictions[0][idx]*100:.2f}%")

    # Hiển thị hình ảnh
    plt.figure(figsize=(6, 6))
    # Chuyển đổi từ BGR sang RGB
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img
    plt.imshow(img_display)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence, predictions[0]


def predict_batch(model, directory, config, max_images=10):
    """Dự đoán lớp cho nhiều hình ảnh trong một thư mục"""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    images = []

    # Tìm tất cả các file hình ảnh trong thư mục
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory, filename))

    # Giới hạn số lượng hình ảnh
    images = images[: min(max_images, len(images))]

    if not images:
        print(f"No images found in {directory}")
        return

    results = []
    plt.figure(figsize=(15, max(2, len(images) // 3 * 2)))

    for i, image_path in enumerate(images):
        # Tiền xử lý hình ảnh
        img = load_and_preprocess_image(
            image_path, target_size=tuple(config["model"]["input_shape"][:2])
        )
        img_array = np.expand_dims(img, axis=0)

        # Dự đoán
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = config["classes"][predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100

        # Lưu kết quả
        results.append(
            {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "confidence": confidence,
            }
        )

        # Hiển thị hình ảnh và dự đoán
        plt.subplot(max(1, len(images) // 3), min(3, len(images)), i + 1)
        img_display = (
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img
        )
        plt.imshow(img_display)
        plt.title(
            f"{os.path.basename(image_path)}\n{predicted_class} ({confidence:.1f}%)",
            fontsize=9,
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return results


def main():
    parser = argparse.ArgumentParser(description="Flower Classification Prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Đường dẫn đến file config",
    )
    parser.add_argument("--image", type=str, help="Đường dẫn đến hình ảnh để dự đoán")
    parser.add_argument(
        "--dir", type=str, help="Đường dẫn đến thư mục chứa nhiều hình ảnh để dự đoán"
    )
    args = parser.parse_args()

    # Đọc cấu hình
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Tải mô hình
    model_path = config["training"]["checkpoint_path"]
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    model = models.load_model(model_path)

    # Thực hiện dự đoán
    if args.image:
        predict_single_image(model, args.image, config)
    elif args.dir:
        predict_batch(model, args.dir, config)
    else:
        print("Please provide either --image or --dir parameter")


if __name__ == "__main__":
    main()
