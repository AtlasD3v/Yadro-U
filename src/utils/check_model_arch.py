import torch
import torchvision

def chech_arch(model: torch.nn.Module):
    for c_name, children in model.named_children():
        print(c_name, "->", children)

def select_model():
    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_large(weights = weights)

    chech_arch(model=model)

# select_model()