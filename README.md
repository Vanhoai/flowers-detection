# 🌸 Flower Classification and Object Detection

A deep learning project for flower classification and object detection using PyTorch and Keras.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Project Overview

This project implements both flower classification and object detection models to identify different flower species and locate them within images. It uses state-of-the-art deep learning techniques implemented with PyTorch and Keras.

## ✨ Features

- **Flower Classification**: Identifies flower species from images with high accuracy
- **Flower Detection**: Locates and identifies flowers in images using object detection models
- **Multi-framework Implementation**: Uses both PyTorch and Keras for model implementation
- **Transfer Learning**: Leverages pre-trained models for faster convergence and better performance
- **Evaluation Metrics**: Comprehensive performance metrics for both classification and detection tasks

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flower-classification-detection.git
cd flower-classification-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

This project uses two datasets:

- **Flower Classification**: [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) with 102 flower categories
- **Flower Detection**: Custom-labeled dataset created from flower images with bounding box annotations

### Dataset Preparation

```bash
# Download and prepare classification dataset
python scripts/prepare_classification_dataset.py

# Download and prepare detection dataset
python scripts/prepare_detection_dataset.py
```

## 📁 Project Structure

```
flower-classification-detection/
├── data/
│   ├── classification/
│   └── detection/
├── models/
│   ├── classification/
│   │   ├── pytorch_models.py
│   │   └── keras_models.py
│   └── detection/
│       ├── pytorch_models.py
│       └── keras_models.py
├── scripts/
│   ├── prepare_classification_dataset.py
│   └── prepare_detection_dataset.py
├── utils/
│   ├── data_utils.py
│   ├── visualization.py
│   └── evaluation.py
├── train.py
├── evaluate.py
├── predict.py
├── app.py
└── requirements.txt
```

## 🧠 Model Architecture

### Classification Models

1. **PyTorch Implementation**:

   - ResNet50 with fine-tuning
   - EfficientNet-B3 with custom classification head

2. **Keras Implementation**:
   - VGG16 with transfer learning
   - MobileNetV2 with custom top layers

### Detection Models

1. **PyTorch Implementation**:

   - Faster R-CNN with ResNet50 backbone
   - YOLO v5 for real-time detection

2. **Keras Implementation**:
   - RetinaNet with ResNet50 backbone
   - SSD with MobileNetV2 feature extractor

## 🔄 Training

### Classification Training

```bash
# Train classification model with PyTorch
python train.py --task classification --framework pytorch --model resnet50 --epochs 30 --batch-size 32

# Train classification model with Keras
python train.py --task classification --framework keras --model mobilenetv2 --epochs 30 --batch-size 32
```

### Detection Training

```bash
# Train detection model with PyTorch
python train.py --task detection --framework pytorch --model faster_rcnn --epochs 50 --batch-size 16

# Train detection model with Keras
python train.py --task detection --framework keras --model retinanet --epochs 50 --batch-size 16
```

## 📏 Evaluation

```bash
# Evaluate classification model
python evaluate.py --task classification --framework pytorch --model resnet50 --weights path/to/weights.pth

# Evaluate detection model
python evaluate.py --task detection --framework keras --model retinanet --weights path/to/weights.h5
```

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Detection Metrics

- Mean Average Precision (mAP)
- Precision-Recall Curve
- IoU (Intersection over Union)
- FPS (Frames Per Second)

## 📋 Usage

### Classification

```python
from models.classification.pytorch_models import ResNet50Classifier
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = ResNet50Classifier(num_classes=102)
model.load_state_dict(torch.load('weights/resnet50_classifier.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = Image.open('path/to/flower.jpg')
input_tensor = transform(image).unsqueeze(0)

# Get prediction
with torch.no_grad():
    output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)

print(f"Predicted flower class: {predicted_idx.item()}")
```

### Detection

```python
from models.detection.pytorch_models import FasterRCNNDetector
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load model
model = FasterRCNNDetector(num_classes=102)
model.load_state_dict(torch.load('weights/faster_rcnn_detector.pth'))
model.eval()

# Prepare image
image = Image.open('path/to/flower_scene.jpg')
input_tensor = transforms.ToTensor()(image).unsqueeze(0)

# Get predictions
with torch.no_grad():
    predictions = model(input_tensor)

# Display results
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
    if score > 0.5:  # Confidence threshold
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"Flower {label}: {score:.2f}", color='white', backgroundcolor='red')

plt.show()
```

## 📊 Results

### Classification Performance

| Model           | Framework | Accuracy | F1 Score | Training Time |
| --------------- | --------- | -------- | -------- | ------------- |
| ResNet50        | PyTorch   | 94.2%    | 0.93     | 3.5h          |
| EfficientNet-B3 | PyTorch   | 95.8%    | 0.95     | 5.2h          |
| VGG16           | Keras     | 92.5%    | 0.92     | 4.1h          |
| MobileNetV2     | Keras     | 93.1%    | 0.93     | 2.8h          |

### Detection Performance

| Model        | Framework | mAP@0.5 | mAP@0.5:0.95 | FPS |
| ------------ | --------- | ------- | ------------ | --- |
| Faster R-CNN | PyTorch   | 0.86    | 0.57         | 8   |
| YOLO v5      | PyTorch   | 0.84    | 0.52         | 45  |
| RetinaNet    | Keras     | 0.85    | 0.56         | 12  |
| SSD          | Keras     | 0.81    | 0.48         | 35  |

## 🚀 Future Improvements

- Implement ensemble methods for better classification accuracy
- Add support for more detection architectures (e.g., YOLO v8, DETR)
- Create a web interface for real-time flower identification
- Expand the dataset to include more flower species
- Implement model quantization for mobile deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
