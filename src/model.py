import torch.nn as nn
from torchvision import models


def create_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """Create a ResNet18 architecture adapted for 32x32 CIFAR-10 images."""
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
