"""
Diane Hamilton

encodes metadata data
    simple multilayer perceptron to train for tabular data
"""

import torch.nn as nn
from constants import *

class MushroomTabularEncoder(nn.Module):
    # input_dim = num features post-preprocessing of tabular data
    def __init__(self, input_dim, out_dim=output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_ftr),
            nn.ReLU(),
            nn.Linear(output_ftr, out_dim)
        )
    def forward(self, x):
        return self.model(x)
