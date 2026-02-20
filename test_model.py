import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os


# Simple CNN architecture (must match Exp 1 exactly)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleCNN(num_classes=1)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")
    
    return model, device


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor


def predict(model, image_path, device):
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor).squeeze()
        score = output.item()
    
    label = 'dog' if score > 0.5 else 'cat'
    confidence = score if score > 0.5 else 1 - score
    
    return label, confidence, score


if __name__ == "__main__":
    # CHANGE THIS to your model path
    MODEL_PATH = r"model\simple_cnn_baseline_exp1_20260217_053749_best.pt"
    
    # CHANGE THIS to your image path
    IMAGE_PATH = r"image1.jpg"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        sys.exit(1)
    
    model, device = load_model(MODEL_PATH)
    label, confidence, raw_score = predict(model, IMAGE_PATH, device)
    
    print(f"\nImage: {IMAGE_PATH}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw score: {raw_score:.4f} (0=cat, 1=dog)")