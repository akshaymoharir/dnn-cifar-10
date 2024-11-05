
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

## Following are only ML related third party libraries
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# Following library utils has utility functions written by myself
from utils import *

# Create training and validation datasets
training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
validation_data = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())

# Create dataloader and provide datasets for loading
batch_size = 32
train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

# Explore dataset
explore_dataset(training_data=training_data ,validation_data=validation_data, 
                train_dataloader=train_dataloader, validation_dataloader=validation_dataloader)


