
# This is a project bring up file.

import numpy as np
import torch

print(f"Torch version:{torch.__version__}")
print(f"Numpy version:{np.__version__}")

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Create training and validation datasets
training_data = datasets.CIFAR10(root="data",train=True,download=True,transform=ToTensor())
validation_data = datasets.CIFAR10(root="data",train=False,download=True,transform=ToTensor())

# Create dataloader and provide datasets for loading
batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

for X,y in validation_dataloader:
    print(f"Shape of X:[N, C, H, W]:{X.shape}")
    print(f"Shape of y:{y.shape}")
    break
