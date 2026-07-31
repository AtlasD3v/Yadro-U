import torch
import torchvision


def build_model(num_classes, trainable_backbone_layers = None):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights = None,
        weights_backbone = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT,
        num_classes = num_classes,
        trainable_backbone_layers = trainable_backbone_layers
    )

    return model


def optim_groups():
    pass