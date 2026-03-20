import copy
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


def training_one_batch(model, X_batch, y_batch, optimizer, loss_fn):
    # forward pass: compute model predictions and the loss
    y_pred = model(X_batch)
    loss = loss_fn(y_pred, y_batch)      
    # backward pass: compute gradients to know how to change weights
    optimizer.zero_grad()
    loss.backward()
    # update weights
    optimizer.step()

def training_one_epoch(model, X_train, y_train, 
                       optimizer, loss, index_where_batches_start):
    for start in index_where_batches_start:
        # Select the batch of data
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        training_one_batch(model, X_batch, y_batch, optimizer, loss)

def evaluation_one_epoch(model, X_val, y_val, loss_fn, 
                         loss_history, best_loss_value, best_weights):
    y_pred = model(X_val)
    loss_value = loss_fn(y_pred, y_val).detach()
    loss_value = float(loss_value)
    loss_history.append(loss_value)
    if loss_value < best_loss_value:
        best_loss_value = loss_value
        best_weights = copy.deepcopy(model.state_dict())
    return best_loss_value, best_weights
    
