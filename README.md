# ASL Alphabet Classification with ConvNeXt

This project implements fine-tuning of ConvNeXt models for American Sign Language (ASL) alphabet classification using the Kaggle ASL alphabet dataset.

## Features

- **ConvNeXt Architecture**: Uses state-of-the-art ConvNeXt models (tiny, small, base variants)
- **Transfer Learning**: Fine-tunes pretrained models for ASL classification
- **Data Augmentation**: Comprehensive augmentation pipeline for better generalization
- **Training Monitoring**: Real-time training progress with loss and accuracy tracking
- **Model Evaluation**: Detailed evaluation with classification reports
- **Inference Script**: Easy-to-use inference for new images

## Dataset

Download the ASL Alphabet dataset from Kaggle:
```bash
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip
```

The dataset should be organized as:
```
asl_alphabet_train/
├── A/
├── B/
├── C/
...
├── Z/
├── del/
├── nothing/
└── space/
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. For Kaggle dataset download:
```bash
pip install kaggle
```

## Usage

### Training

1. Update the `DATA_DIR` variable in `train.py` to point to your dataset directory
2. Run training:
```bash
python train.py
```

### Configuration Options

You can modify these parameters in the main function:

- `BATCH_SIZE`: Batch size for training (default: 32)
- `NUM_EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `MODEL_NAME`: ConvNeXt variant ('convnext_tiny', 'convnext_small', 'convnext_base')

### Inference

Use the trained model for predictions:

```python
from inference import load_model, predict_image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, class_names, label_to_idx = load_model('asl_convnext_complete_model.pth', device)

predicted_class, confidence, probabilities = predict_image(
    model, 'your_image.jpg', class_names, device
)

print(f'Predicted: {predicted_class} (confidence: {confidence:.4f})')
```

## Model Architecture

The ConvNeXt model is modified with:
- Custom classifier head with dropout layers
- 512 hidden units in the intermediate layer
- Dropout rates of 0.3 for regularization

## Training Pipeline

1. **Data Loading**: Loads images and creates train/validation/test splits
2. **Data Augmentation**: Applies random transformations for better generalization
3. **Model Training**: Fine-tunes ConvNeXt with cosine annealing scheduler
4. **Model Evaluation**: Evaluates on test set with detailed metrics
5. **Model Saving**: Saves best model based on validation accuracy

## Expected Results

- Training typically achieves 95%+ accuracy on validation set
- Test accuracy should be around 90-95% depending on the model variant
- Training time: ~30-60 minutes on GPU for 20 epochs

## Files Generated

- `best_asl_convnext_model.pth`: Best model weights during training
- `asl_convnext_complete_model.pth`: Complete model with metadata
- `training_history.png`: Training/validation loss and accuracy plots

## Tips for Better Performance

1. **Increase epochs**: Train for more epochs (30-50) for better convergence
2. **Use larger models**: Try 'convnext_small' or 'convnext_base' for better accuracy
3. **Adjust learning rate**: Use learning rate scheduling or lower initial rates
4. **Data augmentation**: Experiment with different augmentation strategies
5. **Ensemble methods**: Combine multiple models for better predictions

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use smaller model variant
- **Slow training**: Enable num_workers in DataLoader, use GPU if available
- **Poor accuracy**: Check data quality, increase model capacity, or train longer