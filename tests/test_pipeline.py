import pytest
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os

# Add parent directory to path so we can import from test_model.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_model import SimpleCNN, preprocess_image


# ─────────────────────────────────────────────
# 1. Preprocessing Tests
# ─────────────────────────────────────────────

class TestPreprocessImage:

    def test_output_is_tensor(self, tmp_path):
        """preprocess_image should return a torch.Tensor"""
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        result = preprocess_image(str(img_path))
        assert isinstance(result, torch.Tensor), "Output should be a torch.Tensor"

    def test_output_shape(self, tmp_path):
        """Output tensor should be shape (1, 3, 224, 224)"""
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        result = preprocess_image(str(img_path))
        assert result.shape == (1, 3, 224, 224), f"Expected (1,3,224,224), got {result.shape}"

    def test_handles_non_square_image(self, tmp_path):
        """Should resize any image shape to 224x224"""
        img = Image.fromarray(np.uint8(np.random.rand(300, 150, 3) * 255))
        img_path = tmp_path / "nonsquare.jpg"
        img.save(img_path)

        result = preprocess_image(str(img_path))
        assert result.shape == (1, 3, 224, 224)

    def test_normalized_values(self, tmp_path):
        """Normalized tensor should not be in raw [0,255] range"""
        img = Image.fromarray(np.uint8(np.ones((100, 100, 3)) * 200))
        img_path = tmp_path / "bright.jpg"
        img.save(img_path)

        result = preprocess_image(str(img_path))
        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert result.max().item() < 10, "Values seem un-normalized"
        assert result.min().item() > -10, "Values seem un-normalized"


# ─────────────────────────────────────────────
# 2. Model / Inference Utility Tests
# ─────────────────────────────────────────────

class TestSimpleCNN:

    def test_model_instantiation(self):
        """Model should instantiate without errors"""
        model = SimpleCNN(num_classes=1)
        assert model is not None

    def test_forward_pass_output_shape(self):
        """Forward pass should return shape (batch_size, 1)"""
        model = SimpleCNN(num_classes=1)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 1), f"Expected (1,1), got {output.shape}"

    def test_output_in_valid_range(self):
        """Sigmoid output should be between 0 and 1"""
        model = SimpleCNN(num_classes=1)
        model.eval()
        dummy_input = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert (output >= 0).all() and (output <= 1).all(), "Sigmoid output must be in [0, 1]"

    def test_batch_inference(self):
        """Model should handle batch input correctly"""
        model = SimpleCNN(num_classes=1)
        model.eval()
        batch = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            output = model(batch)
        assert output.shape[0] == 8, "Batch size mismatch in output"

    def test_model_is_in_eval_mode(self):
        """Model should support eval mode (dropout disabled)"""
        model = SimpleCNN(num_classes=1)
        model.eval()
        assert not model.training, "Model should be in eval mode"
