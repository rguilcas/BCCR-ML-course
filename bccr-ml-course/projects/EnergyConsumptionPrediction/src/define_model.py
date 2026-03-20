from torch import nn
import torch.nn.functional as F

class MultiLayerPerceptron3layers(nn.Module):
    def __init__(self, 
                 number_input_features=3, 
                 number_of_predictions=1):
        super().__init__()
        self.layer1 = nn.Linear(number_input_features, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, number_of_predictions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
