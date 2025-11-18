import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


def build_model(num_classes: int = 2) -> nn.Module:
    """Create a MobileNetV2-based classifier for fatigue-related tasks.

    Parameters
    ----------
    num_classes: int
        Number of output classes (e.g. 2 for binary classification).
    """
    # Load imagenet-pretrained MobileNetV2
    model = mobilenet_v2(weights="IMAGENET1K_V1")

    # Replace the classifier to match our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
