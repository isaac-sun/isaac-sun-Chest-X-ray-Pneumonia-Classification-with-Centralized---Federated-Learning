from typing import Dict

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline for binary chest X-ray classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def build_model(model_name: str, pretrained: bool = False):
    """Create model instance by name and adapt final layer for single-logit output."""
    model_name = model_name.lower()

    if model_name == "simple_cnn":
        return SimpleCNN()

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


def get_target_layer(model_name: str, model):
    """Return the conv layer used by Grad-CAM based on model architecture."""
    if model_name.lower() == "resnet18":
        return model.layer4[-1].conv2
    if model_name.lower() == "simple_cnn":
        return model.features[12]
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")
