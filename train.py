import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import timm
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ASLDataset(Dataset):
    """Custom dataset for ASL alphabet images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_dataset(data_dir):
    """Load ASL dataset and create train/test splits"""
    image_paths = []
    labels = []
    class_names = []
    
    # Get all class directories
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            
    # Create label mapping
    label_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Load all images and labels
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(label_to_idx[class_name])
    
    return image_paths, labels, class_names, label_to_idx

def get_transforms():
    """Define data augmentation and preprocessing transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class ConvNeXtASL(nn.Module):
    """ConvNeXt model for ASL classification"""
    
    def __init__(self, num_classes, model_name='convnext_tiny', pretrained=True):
        super(ConvNeXtASL, self).__init__()
        
        # Load pretrained ConvNeXt model
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the number of features from the classifier
        num_features = self.backbone.head.fc.in_features
        
        # Replace the classifier
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Training loop for the model"""
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        train_pbar = tqdm(train_loader, desc='Training', leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct_predictions / val_total_predictions:.2f}%'
                })
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct_predictions / val_total_predictions
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_asl_convnext_model.pth')
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
        
        # Step scheduler
        scheduler.step()
    
    return train_losses, train_accuracies, val_losses, val_accuracies, best_model_state

def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader, class_names, device):
    """Evaluate model on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'\nTest Accuracy: {accuracy * 100:.2f}%')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return accuracy, all_predictions, all_labels

def main():
    """Main training pipeline"""
    
    # Configuration
    DATA_DIR = 'asl_alphabet_train'  # Change this to your dataset path
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    MODEL_NAME = 'convnext_tiny'  # Options: convnext_tiny, convnext_small, convnext_base
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    print('Loading dataset...')
    image_paths, labels, class_names, label_to_idx = load_dataset(DATA_DIR)
    print(f'Found {len(image_paths)} images across {len(class_names)} classes')
    print(f'Classes: {class_names}')
    
    # Split dataset
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f'Train samples: {len(train_paths)}')
    print(f'Validation samples: {len(val_paths)}')
    print(f'Test samples: {len(test_paths)}')
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = ASLDataset(train_paths, train_labels, train_transform)
    val_dataset = ASLDataset(val_paths, val_labels, val_transform)
    test_dataset = ASLDataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    print(f'\nCreating {MODEL_NAME} model...')
    model = ConvNeXtASL(num_classes=len(class_names), model_name=MODEL_NAME, pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Train model
    print('\nStarting training...')
    train_losses, train_accuracies, val_losses, val_accuracies, best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device
    )
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, class_names, device)
    
    # Save model and metadata
    torch.save({
        'model_state_dict': best_model_state,
        'class_names': class_names,
        'label_to_idx': label_to_idx,
        'model_name': MODEL_NAME,
        'test_accuracy': test_accuracy
    }, 'asl_convnext_complete_model.pth')
    
    print(f'\nTraining completed! Best model saved with test accuracy: {test_accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()