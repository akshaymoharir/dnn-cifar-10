

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from configs import BATCH_SIZE


# Required to visualize from container running on mac
matplotlib.use('Agg')

# Activation function (sigmoid) and its derivative
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

def relu(z):
    return max(0,z)

def relu_derivative(z):
    relu_derivative = 0
    if z > 0:
        relu_derivative = 1
    return relu_derivative

def explore_dataset():

    training_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    validation_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    
    print(f"\nClasses in CIFAR10 dataset are:\n{training_data.class_to_idx}\n")
    # Print number of examples in training and validation data
    print(f"Number of training examples:{training_data.__len__()}")
    print(f"Number of validation examples:{validation_data.__len__()}")
    # Print shape of data
    print(f"Shape of training data:{training_data.data.shape}")

    # Explore one random training example
    rng = np.random.default_rng()
    training_data_random_idx = rng.integers(low=0,high=training_data.__len__())
    print(f"Random example chosen:{training_data_random_idx}")
    num_examples, height, width, num_channels = training_data.data.shape
    # Print size of each training sample
    print(f"Shape of one training image:{training_data.data[training_data_random_idx].shape}")
    print(f"Number of examples:{num_examples}\nHeight:{height}\nWidth:{width}\nNumber of channels:{num_channels}")
    # Plot random example chosen
    sample_image, sample_label = training_data[training_data_random_idx]
    sample_image = np.transpose(sample_image, (1, 2, 0))
    #plt.imshow(sample_image)
    plt.title(training_data.classes[sample_label]) 
    plt.imsave("sample_image.png", sample_image)

    # Plot random examples from training data for each class
    # get some random training images
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    sample_grid = torchvision.utils.make_grid(images)
    sample_grid = np.transpose(sample_grid.numpy(), (1, 2, 0))
    plt.imsave("some_more_samples.png",sample_grid)
    