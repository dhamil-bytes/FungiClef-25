"""
Diane Hamilton

encodes image data
utilizes pretrained resnet18
reason:
    despite resnet50 being trained on mostly imagenet data, its architecture is dense
    and is able to pickup texture and grain in a photo faster. its cnn architecture
    is also desirable for our purposes. 
"""

import torch.nn as nn
from constants import *
from torchvision.models import resnet50

class MushroomImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet50(pretrained=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, output_dim)

        self.model = backbone

    # shape: (batch_size, output_dim)
    def forward(self, x):
        return self.model(x)  