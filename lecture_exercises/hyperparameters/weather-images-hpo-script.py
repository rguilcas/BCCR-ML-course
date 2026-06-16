from PIL import Image

# We rely on several libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
import sys
import time
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# We load in some code we've defined elsewhere, in the src folder
weather_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
    +'/projects/WeatherImagesClassification'
sys.path.append(weather_dir)
print(sys.path)

from src.weather_helpers import (
    WeatherDataset, filepath, metapath, reset_seeds, plot_learning_curves, get_class_name
)

from src.define_model import Net

from src.train_and_evaluate import evaluate_model, evaluate_wrapper, train_model

# We have some objects that hold our data.
# The data split has already been done for us
# (if you're curious, you can look in src/preprocess_data.py)
start = time.time()
D = 32 # data dimensionality.
# We keep D constant to avoid having to reload the data every time we train a model.
trainset = WeatherDataset(filepath, metapath, subset='train', D=D)
valset = WeatherDataset(filepath, metapath, subset='val', D=D)
print(f'Data loading took {np.round(time.time()-start, 4)} seconds.')

# Debugging: Make sure we have a GPU available 
print(torch.cuda.is_available())
device = torch.device('cuda')

project_id = "my-panda-sweep-8" # Pock an identifier

def try_settings():
    with wandb.init(project=project_id) as run:
        B = 32
        lr = run.config.lr
        # Pytorch provides data loaders, which helpfully groups our data into batches
        # Here we also choose to shuffle the training data, but not the validation data
        train_dataloader = DataLoader(trainset, batch_size=B, shuffle=True)
        val_dataloader = DataLoader(valset, batch_size=B, shuffle=False)
        
        reset_seeds(1) # Make things reproducible
        net = Net(D=D, C2=run.config.C2, C3=run.config.C3,
                  L1=2**run.config.L1_expo, L2=2**run.config.L2_expo) # Set up our model
        net.to(device) # Send it to the gpu
        # Let's choose some hyperparameters:
        criterion = nn.CrossEntropyLoss()
        if run.config.optimizer == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        elif run.config.optimizer == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=lr)
        elif run.config.optimizer == 'AdamW':
            optimizer = optim.AdamW(net.parameters(), lr=lr)
        else:
            raise ValueError
        
        # Train our model
        epochs =  20
        train_curve, val_curve = train_model(
            net, val_dataloader, train_dataloader, device, epochs, optimizer, criterion)
        for val in val_curve:
            run.log({'score': val})


sweep_configuration = {
    "method": "random", # Alternatives are "grid", "random", "bayes"
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "lr": {"max": 0.01, "min": 0.000001, "distribution": "log_uniform_values"},
        "C2": {"max": 32, "min": 3, "distribution": "int_uniform"},
        "C3": {"max": 32, "min": 3, "distribution": "int_uniform"},
        "L1_expo": {"max": 10, "min": 1, "distribution": "int_uniform"},
        "L2_expo": {"max": 10, "min": 1, "distribution": "int_uniform"},
        "optimizer": {"values": ["SGD", "Adam", "AdamW"], "distribution": "categorical"},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_id)

wandb.agent(sweep_id, function=try_settings, count=64)
