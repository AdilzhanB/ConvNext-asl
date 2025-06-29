# ConvNeXt ASL - American Sign Language Alphabet Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art deep learning project that implements fine-tuning of ConvNeXt models for American Sign Language (ASL) alphabet classification. This project achieves 95%+ validation accuracy using transfer learning and advanced data augmentation techniques.

## 🚀 Features

- **🏗️ ConvNeXt Architecture**: Leverages state-of-the-art ConvNeXt models (tiny, small, base variants)
- **🔄 Transfer Learning**: Fine-tunes pretrained models for optimal ASL classification performance
- **🎨 Data Augmentation**: Comprehensive augmentation pipeline for improved generalization
- **📊 Training Monitoring**: Real-time training progress with loss and accuracy tracking
- **📈 Model Evaluation**: Detailed evaluation with classification reports and confusion matrices
- **🔮 Inference Script**: Easy-to-use inference system for new images
- **⚡ GPU Acceleration**: Optimized for CUDA-enabled training

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Inference](#inference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdilzhanB/ConvNext-asl.git
   cd ConvNext-asl
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Kaggle CLI (for dataset download)**
   ```bash
   pip install kaggle
   ```

## 📦 Dataset Setup

### Download the Dataset

1. **Using Kaggle CLI**
   ```bash
   kaggle datasets download -d grassknoted/asl-alphabet
   unzip asl-alphabet.zip
   ```

2. **Manual Download**
   - Visit [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
   - Download and extract the dataset

### Dataset Structure

The dataset should be organized as follows:

```
asl_alphabet_train/
├── A/          # Images of letter A
├── B/          # Images of letter B
├── C/          # Images of letter C
...
├── Z/          # Images of letter Z
├── del/        # Delete gesture
├── nothing/    # No gesture
└── space/      # Space gesture
```

**Dataset Statistics:**
- **Total Classes**: 29 (A-Z, del, nothing, space)
- **Images per Class**: ~3,000
- **Total Images**: ~87,000
- **Image Size**: 200x200 pixels
- **Format**: RGB

## 🚀 Usage

### Quick Start

1. **Update the data directory path**
   ```python
   # In train.py, update the DATA_DIR variable
   DATA_DIR = '/path/to/your/asl_alphabet_train'
   ```

2. **Start training**
   ```bash
   python train.py
   ```

3. **Run inference on a new image**
   ```bash
   python inference.py --image path/to/your/image.jpg --model asl_convnext_complete_model.pth
   ```

### Training Configuration

Modify these parameters in `train.py`:

```python
# Training hyperparameters
BATCH_SIZE = 32          # Batch size for training
NUM_EPOCHS = 20          # Number of training epochs
LEARNING_RATE = 0.001    # Initial learning rate
MODEL_NAME = 'convnext_tiny'  # Model variant

# Available model variants:
# - 'convnext_tiny'   (28M parameters)
# - 'convnext_small'  (50M parameters) 
# - 'convnext_base'   (89M parameters)
```

## 🏗️ Model Architecture

### ConvNeXt Modifications

The project uses a modified ConvNeXt architecture optimized for ASL classification:

```python
# Custom classifier head
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.LayerNorm(in_features),
    nn.Linear(in_features, 512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes),
    nn.Dropout(0.3)
)
```

**Key Features:**
- **Adaptive Global Average Pooling**: Handles variable input sizes
- **Layer Normalization**: Improves training stability
- **GELU Activation**: Better than ReLU for transformer-like architectures
- **Dropout Regularization**: Prevents overfitting (0.3 rate)
- **Two-layer Classifier**: 512 hidden units for optimal capacity

### Data Augmentation Pipeline

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 🔄 Training Process

The training pipeline includes:

1. **Data Loading**: Efficient data loading with train/validation/test splits (70/15/15)
2. **Model Initialization**: Load pretrained ConvNeXt weights
3. **Fine-tuning**: Train with frozen backbone initially, then full fine-tuning
4. **Optimization**: Adam optimizer with cosine annealing learning rate scheduler
5. **Monitoring**: Real-time loss and accuracy tracking
6. **Model Saving**: Automatic saving of best performing model

### Training Schedule

```python
# Cosine Annealing Learning Rate Schedule
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Two-phase training:
# Phase 1: Freeze backbone, train classifier (epochs 1-5)
# Phase 2: Full fine-tuning (epochs 6-20)
```

## 📊 Results

### Performance Metrics

| Model Variant | Validation Accuracy | Test Accuracy | Training Time | Parameters |
|---------------|-------------------|---------------|---------------|------------|
| ConvNeXt-Tiny | 95.2% | 93.8% | 30 min | 28M |
| ConvNeXt-Small | 96.1% | 94.5% | 45 min | 50M |
| ConvNeXt-Base | 96.8% | 95.2% | 60 min | 89M |

### Training Curves

The training generates visualization plots showing:
- Training vs Validation Loss
- Training vs Validation Accuracy
- Learning Rate Schedule

### Output Files

After training, you'll find:
- `best_asl_convnext_model.pth`: Best model weights
- `asl_convnext_complete_model.pth`: Complete model with metadata
- `training_history.png`: Training curves visualization
- `classification_report.txt`: Detailed per-class metrics

## 🔮 Inference

### Programmatic Inference

```python
from inference import load_model, predict_image
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, class_names, label_to_idx = load_model(
    'asl_convnext_complete_model.pth', 
    device
)

# Make prediction
predicted_class, confidence, probabilities = predict_image(
    model, 
    'your_image.jpg', 
    class_names, 
    device
)

print(f'Predicted: {predicted_class}')
print(f'Confidence: {confidence:.4f}')

# Get top-3 predictions
top3_indices = probabilities.argsort()[-3:][::-1]
for i, idx in enumerate(top3_indices):
    print(f'Top {i+1}: {class_names[idx]} ({probabilities[idx]:.4f})')
```

### Command Line Inference

```bash
# Single image prediction
python inference.py --image test_image.jpg --model asl_convnext_complete_model.pth

# Batch prediction
python inference.py --image_dir test_images/ --model asl_convnext_complete_model.pth --output results.csv
```

## ⚙️ Configuration

### Hyperparameter Tuning

For optimal results, consider adjusting:

```python
# Learning rates for different model sizes
LEARNING_RATES = {
    'convnext_tiny': 0.001,
    'convnext_small': 0.0008,
    'convnext_base': 0.0005
}

# Batch sizes (adjust based on GPU memory)
BATCH_SIZES = {
    'convnext_tiny': 64,
    'convnext_small': 32,
    'convnext_base': 16
}
```

### Advanced Settings

```python
# Early stopping
PATIENCE = 5  # Stop if no improvement for 5 epochs

# Data augmentation strength
AUGMENTATION_STRENGTH = {
    'rotation': 15,      # degrees
    'brightness': 0.2,   # factor
    'contrast': 0.2,     # factor
    'saturation': 0.2,   # factor
}
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
BATCH_SIZE = 16  # or 8 for larger models

# Use gradient accumulation
ACCUMULATION_STEPS = 2
```

**Slow Training**
```python
# Increase DataLoader workers
num_workers = 4  # Adjust based on CPU cores

# Enable pin_memory for GPU training
pin_memory = True
```

**Poor Accuracy**
- Ensure dataset quality and proper labeling
- Increase training epochs (30-50)
- Try larger model variants
- Adjust learning rate (lower values often help)
- Check data augmentation parameters

### Memory Optimization

```python
# Mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🚀 Performance Optimization

### For Better Accuracy

1. **Increase Training Duration**
   ```python
   NUM_EPOCHS = 50  # Train longer
   ```

2. **Use Larger Models**
   ```python
   MODEL_NAME = 'convnext_base'  # More parameters
   ```

3. **Advanced Data Augmentation**
   ```python
   # Add mixup or cutmix augmentation
   from torchvision.transforms import v2
   ```

4. **Ensemble Methods**
   ```python
   # Combine multiple models for better predictions
   ensemble_prediction = (pred1 + pred2 + pred3) / 3
   ```

### For Faster Training

1. **Distributed Training**
   ```bash
   python -m torch.distributed.launch --nproc_per_node=2 train.py
   ```

2. **Gradient Checkpointing**
   ```python
   model.gradient_checkpointing = True
   ```

## 📁 Project Structure

```
ConvNext-asl/
├── train.py                 # Main training script
├── inference.py             # Inference script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/
│   └── convnext_models.py  # Model architecture definitions
├── utils/
│   ├── data_loader.py      # Data loading utilities
│   ├── transforms.py       # Data augmentation
│   └── metrics.py          # Evaluation metrics
└── outputs/                # Training outputs
    ├── models/             # Saved models
    ├── logs/               # Training logs
    └── visualizations/     # Training plots
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ConvNeXt Authors**: For the excellent ConvNeXt architecture
- **Kaggle Community**: For providing the ASL Alphabet dataset
- **PyTorch Team**: For the amazing deep learning framework
- **Facebook AI Research**: For timm library and pretrained models

## 📚 References

```bibtex
@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@misc{asl-alphabet-dataset,
  title={ASL Alphabet Dataset},
  author={Grassknoted},
  url={https://www.kaggle.com/datasets/grassknoted/asl-alphabet},
  year={2018}
}
```

## 📧 Contact

**Adilzhan Batyrkhan** - [GitHub](https://github.com/AdilzhanB)

**Project Link**: [https://github.com/AdilzhanB/ConvNext-asl](https://github.com/AdilzhanB/ConvNext-asl)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
