import pytest
import torch
import io
from PIL import Image
from app import SimpleCNN, preprocess, predict_image

# Test 1: Preprocessing function
def test_preprocess_output_shape():
    img = Image.new('RGB', (300, 300))
    tensor = preprocess(img)
    assert tensor.shape == (3, 224, 224), "Preprocessed tensor shape mismatch"

def test_preprocess_normalization():
    img = Image.new('RGB', (224, 224))
    tensor = preprocess(img)
    assert tensor.min() < 0, "Normalization failed - values should go negative"

# Test 2: Model inference function
def test_model_output_range():
    model = SimpleCNN(num_classes=1)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input).squeeze().item()
    assert 0.0 <= output <= 1.0, "Model output should be between 0 and 1"

def test_model_architecture():
    model = SimpleCNN(num_classes=1)
    assert isinstance(model.conv_layers, torch.nn.Sequential)
    assert isinstance(model.fc_layers, torch.nn.Sequential)

# Test 3: Health check endpoint
def test_health_endpoint():
    from app import app
    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

# Test 4: Predict endpoint with dummy image
def test_predict_endpoint():
    from app import app
    client = app.test_client()
    img = Image.new('RGB', (224, 224))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    response = client.post('/predict', data={
        'file': (buf, 'test.jpg')
    }, content_type='multipart/form-data')
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert data['prediction'] in ['cat', 'dog']