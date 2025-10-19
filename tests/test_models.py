# test_models.py
from models.lightweight_cnn import LightweightCNN
from models.medium_lstm import MediumLSTM
from models.full_model import FullCNNLSTM
import torch

def test_models():
    batch_size = 32
    timesteps = 30
    features = 14
    
    x = torch.randn(batch_size, timesteps, features)
    
    # Test lightweight CNN
    model_light = LightweightCNN(input_channels=features, num_classes=2)
    out = model_light(x)
    print(f"Lightweight CNN: {model_light.count_parameters()} params, output shape: {out.shape}")
    
    # Test medium LSTM
    model_medium = MediumLSTM(input_size=features, num_classes=2)
    out = model_medium(x)
    print(f"Medium LSTM: {sum(p.numel() for p in model_medium.parameters())} params, output shape: {out.shape}")
    
    # Test full model
    model_full = FullCNNLSTM(input_channels=features, num_classes=2)
    out = model_full(x)
    print(f"Full CNN-LSTM: {sum(p.numel() for p in model_full.parameters())} params, output shape: {out.shape}")
    
    print("\nAll models working correctly!")

if __name__ == "__main__":
    test_models()