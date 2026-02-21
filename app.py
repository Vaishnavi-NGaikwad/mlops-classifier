from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import io
import base64
import time
import logging
from collections import deque
from datetime import datetime

# ============================================
# M5: Logging Setup
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler('app.log')  # File
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# M5: In-Memory Metrics
# ============================================
metrics = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_requests': 0,
    'total_latency_ms': 0.0,
    'prediction_log': deque(maxlen=100)  # Store last 100 predictions
}

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
MODEL_PATH = 'model/simple_cnn_baseline_exp1_20260217_053749_best.pt'

try:
    model = SimpleCNN(num_classes=1)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully from {MODEL_PATH} on {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
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
    start_time = time.time()
    metrics['total_requests'] += 1

    try:
        if model is None:
            metrics['failed_requests'] += 1
            logger.error("Prediction request failed: model not loaded")
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        if 'file' not in request.files:
            metrics['failed_requests'] += 1
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            metrics['failed_requests'] += 1
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            metrics['failed_requests'] += 1
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        score = predict_image(image)

        predicted_class = 'dog' if score > 0.5 else 'cat'
        confidence = score if score > 0.5 else 1 - score

        # M5: Calculate latency and log prediction
        latency_ms = (time.time() - start_time) * 1000
        metrics['total_latency_ms'] += latency_ms
        metrics['successful_predictions'] += 1

        # M5: Store prediction in log (simulated true label for demo)
        metrics['prediction_log'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'filename': file.filename,
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'raw_score': round(score, 4),
            'latency_ms': round(latency_ms, 2)
        })

        logger.info(
            f"PREDICT | file={file.filename} | "
            f"prediction={predicted_class} | "
            f"confidence={round(confidence * 100, 2)}% | "
            f"latency={round(latency_ms, 2)}ms"
        )

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'label': predicted_class.upper(),
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'cat': round((1 - score) * 100, 2),
                'dog': round(score * 100, 2)
            },
            'latency_ms': round(latency_ms, 2),
            'message': f'This pet is a {predicted_class.upper()} with {round(confidence * 100, 2)}% confidence'
        }), 200

    except Exception as e:
        metrics['failed_requests'] += 1
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"PREDICT ERROR | latency={round(latency_ms, 2)}ms | error={str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


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


# ============================================
# M5: ENDPOINT 5 - Metrics
# ============================================
@app.route('/metrics', methods=['GET'])
def get_metrics():
    avg_latency = (
        metrics['total_latency_ms'] / metrics['successful_predictions']
        if metrics['successful_predictions'] > 0 else 0
    )

    return jsonify({
        'total_requests': metrics['total_requests'],
        'successful_predictions': metrics['successful_predictions'],
        'failed_requests': metrics['failed_requests'],
        'average_latency_ms': round(avg_latency, 2),
        'total_latency_ms': round(metrics['total_latency_ms'], 2)
    }), 200


# ============================================
# M5: ENDPOINT 6 - Prediction Log (Model Performance Tracking)
# ============================================
@app.route('/prediction-log', methods=['GET'])
def get_prediction_log():
    log = list(metrics['prediction_log'])
    total = len(log)
    dog_count = sum(1 for p in log if p['prediction'] == 'dog')
    cat_count = total - dog_count
    avg_confidence = (
        sum(p['confidence'] for p in log) / total if total > 0 else 0
    )

    return jsonify({
        'total_predictions': total,
        'dog_predictions': dog_count,
        'cat_predictions': cat_count,
        'average_confidence': round(avg_confidence, 2),
        'recent_predictions': log[-10:]  # Last 10
    }), 200


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)

    logger.info("="*60)
    logger.info("Cat vs Dog Classifier API - Starting")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info("Endpoints: /, /health, /predict, /metrics, /prediction-log")
    logger.info("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)