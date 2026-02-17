from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Simple CNN architecture (must match Exp 1 training code)
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


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'model\\simple_cnn_baseline_exp1_20260217_053749_best.pt'  # CHANGE THIS

try:
    model = SimpleCNN(num_classes=1)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_image(image):
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor).squeeze()
        score = output.item()
    
    return score


# ============================================
# ENDPOINT 1: Home Page
# ============================================
@app.route('/')
def home():
    return render_template('index.html')


# ============================================
# ENDPOINT 2: Health Check
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    if model is not None:
        return jsonify({
            'status': 'healthy',
            'message': 'Cat vs Dog Classifier API is running',
            'model_loaded': True,
            'framework': 'PyTorch',
            'device': str(device),
            'service': 'Pet Adoption Platform - Image Classification'
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded',
            'model_loaded': False
        }), 500


# ============================================
# ENDPOINT 3: Prediction
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500

        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'
            }), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        score = predict_image(image)

        predicted_class = 'dog' if score > 0.5 else 'cat'
        confidence = score if score > 0.5 else 1 - score

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'label': predicted_class.upper(),
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'cat': round((1 - score) * 100, 2),
                'dog': round(score * 100, 2)
            },
            'message': f'This pet is a {predicted_class.upper()} with {round(confidence * 100, 2)}% confidence'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================
# ENDPOINT 4: Prediction UI
# ============================================
@app.route('/predict-ui', methods=['POST'])
def predict_ui():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        score = predict_image(image)

        predicted_class = 'dog' if score > 0.5 else 'cat'
        confidence = score if score > 0.5 else 1 - score
        emoji = '🐕' if predicted_class == 'dog' else '🐱'

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'emoji': emoji,
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'cat': round((1 - score) * 100, 2),
                'dog': round(score * 100, 2)
            },
            'image': f'data:image/png;base64,{img_str}'
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    
    print("\n" + "="*60)
    print("Cat vs Dog Classifier API")
    print("Pet Adoption Platform - Image Classification Service")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print("\nAPI Endpoints:")
    print("   Home UI:       http://localhost:5000/")
    print("   Health Check:  http://localhost:5000/health")
    print("   Prediction:    http://localhost:5000/predict")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)