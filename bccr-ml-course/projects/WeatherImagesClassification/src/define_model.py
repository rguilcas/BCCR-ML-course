# Based on example in
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 11 # The number of output classes we've got

class Net(nn.Module):
    # We will start with a small convolutional neural network (CNN) for this project.
    # CNNs are especially suited to computer vision tasks such as our image classification
    # We have two convolutional layers, and afterwards three fully-connected layers,
    # where we gradually shift from (16*5*5=400) to 11 variables
    # Each class has a different output feature, and we then compare which is highest
    # You do not need to understand everything that happens in this model,
    # but you might want to experiment with more or larger layers!
    
    def __init__(self, C1=3, C2=6, C3=16, P=2, kernel_size=5, L1=120, L2=84, D=32):
        # C1, C2, C3: channel counts
        # P: pooling size
        # L1, L2, L3: flat data lengths
        super().__init__()
        self.conv1 = nn.Conv2d(C1, C2, kernel_size)
        self.pool = nn.MaxPool2d(P, P)
        self.conv2 = nn.Conv2d(C2, C3, kernel_size)
        # Calculate how big the images will be after two layers of convolutions and max pooling
        D_after_one_conv_maxpool = (D - (kernel_size-1))//P
        D_after_two_conv_maxpool = (D_after_one_conv_maxpool - (kernel_size-1))//P
        self.fc1 = nn.Linear(C3 * D_after_two_conv_maxpool * D_after_two_conv_maxpool, L1)
        self.fc2 = nn.Linear(L1, L2)
        self.fc3 = nn.Linear(L2, num_classes)

    def forward(self, x):
        # x has dim C1 x 32 x 32 (or C1 x D x D if you've changed this in WeatherDataset)
        x = self.pool(F.relu(self.conv1(x)))
        # x has dim C2 x 14 x 14
        x = self.pool(F.relu(self.conv2(x)))
        # x has dim C3 x 5 x 5
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
