from torch import nn
import torch.nn.functional as F

class MultiLayerPerceptron3layers(nn.Module):
    def __init__(self, 
                 number_input_features=3, 
                 number_of_predictions=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(number_input_features, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, number_of_predictions)
        
    def forward(self, x):
        # We need to flatten the data
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_height=5):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.pool_layer = nn.MaxPool2d(kernel_size=2,stride=2)
        if input_height == 5:
            self.linear_size = 20 
        elif input_height == 16:
            self.linear_size = 20*4*4
        self.flatten = nn.Flatten()
        self.linear_layer1 = nn.Linear(self.linear_size, 16)
        self.linear_layer2 = nn.Linear(16, 16)
        self.linear_layer3 = nn.Linear(16, 1)
        
        
    def forward(self, x):
        x = self.pool_layer(F.relu(self.conv_layer1(x)))
        x = self.pool_layer(F.relu(self.conv_layer2(x)))
        x = self.flatten(x)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        x = self.linear_layer3(x)
        
        return x
