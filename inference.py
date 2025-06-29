import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

class ConvNeXtASL(nn.Module):
    """ConvNeXt model for ASL classification - same as training script"""
    
    def __init__(self, num_classes, model_name='convnext_tiny', pretrained=True):
        super(ConvNeXtASL, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        num_features = self.backbone.head.fc.in_features
        
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ConvNeXtASL(
        num_classes=len(checkpoint['class_names']),
        model_name=checkpoint['model_name'],
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['class_names'], checkpoint['label_to_idx']

def predict_image(model, image_path, class_names, device):
    """Predict ASL sign for a single image"""
    
    # Define preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
    
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, probabilities.cpu().numpy()

def main():
    """Example inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, class_names, label_to_idx = load_model('asl_convnext_complete_model.pth', device)
    
    # Example prediction
    image_path = 'path_to_your_test_image.jpg'  # Change this to your image path
    
    try:
        predicted_class, confidence, probabilities = predict_image(
            model, image_path, class_names, device
        )
        
        print(f'Predicted ASL sign: {predicted_class}')
        print(f'Confidence: {confidence:.4f}')
        
        # Show top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        print('\nTop 3 predictions:')
        for i, idx in enumerate(top3_indices):
            print(f'{i+1}. {class_names[idx]}: {probabilities[idx]:.4f}')
            
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()