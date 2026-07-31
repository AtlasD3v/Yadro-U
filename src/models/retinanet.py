import torch
import torchvision

def build_model(num_classes, trainable_backbone_layers = None):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights = None,
        num_classes = num_classes,
        weights_backbone = torchvision.models.ResNet50_Weights.DEFAULT,
        trainable_backbone_layers = trainable_backbone_layers
    )
    return model
