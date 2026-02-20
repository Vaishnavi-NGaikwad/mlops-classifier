import mlflow
import mlflow.pytorch
import torch
from app import SimpleCNN

mlflow.set_experiment("mlops-image-classifier")  # CHANGE THIS TO YOUR EXPERIMENT NAME

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "SimpleCNN")
    mlflow.log_param("num_classes", 1)
    mlflow.log_param("input_size", 224)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("dropout", 0.5)

    # Log metrics (use your actual values from training)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("val_loss", 0.21)

    # Log the model artifact
    mlflow.log_artifact("model/simple_cnn_baseline_exp1_20260217_053749_best.pt")

    print("Run logged successfully!")