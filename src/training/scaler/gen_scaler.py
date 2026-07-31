import torch

def build_scaler(device: str):
    return torch.amp.GradScaler(device=device)