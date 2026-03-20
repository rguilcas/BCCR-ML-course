import numpy as np
import time
import torch

def evaluate_model(model, loader, device, return_lists = False):
    # Evaluate the model on a data subset.
    # The interface with the data is provided by the data loader 
    total = len(loader.dataset) # Number of example images
    correct = 0 # How many we've predicted correctly so far. At the start, that's zero
    if return_lists:
        # Sometimes we're interested in comparing the predicted with the true labels,
        # e.g. to calculate a confusion matrix or for debugging.
        predicted_list = []
        labels_list = []
    with torch.no_grad(): # Tells torch that we're not currently training
        for data in loader: # Iterate through all the examples
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(device))
            # the class with the highest value is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            # Count number of matching labels
            correct += (predicted.to('cpu') == labels).sum().item()
            if return_lists:
                predicted_list.append(predicted.to('cpu').numpy())
                labels_list.append(labels.numpy())
    
    acc = np.round(100 * correct / total, 4) # Accuracy in percentage
    if return_lists:
        return acc, np.hstack(predicted_list), np.hstack(labels_list)
    return acc

def evaluate_wrapper(net, val_dataloader, train_dataloader, device, val_curve, train_curve):
    # A function to evaluate the current state of the model on the training and validation data
    val_acc = evaluate_model(net, val_dataloader, device)
    print(f'Accuracy of the network on the val images: {val_acc} %')
    val_curve.append(val_acc)
    train_acc = evaluate_model(net, train_dataloader, device)
    print(f'Accuracy of the network on the train images: {train_acc} %')
    train_curve.append(train_acc)
    # We don't need to return the lists, due to how Python passes them to the function

def train_model(net, val_dataloader, train_dataloader, device, epochs,
               optimizer, criterion):
    # Based on example in
    # https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # Function to train our model, going through the training data several times
    # (each pass through the training data is called an "epoch", and we need to
    # decide how many times we want to do that). 
    
    start_t = time.time() # We'll track how long it takes

    train_curve, val_curve = [], [] # Used to keep track of performance as the model learns

    # Check what the performance is at the start
    evaluate_wrapper(net, val_dataloader, train_dataloader, device, val_curve, train_curve)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f'Epoch: \t{epoch}')
    
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # Iterate over the training examples
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # Send our image to the gpu, and run it through the model
            outputs = net(inputs.to(device))
            # Get our predicted class to the cpu, and compare it to the true label
            loss = criterion(outputs.to('cpu'), labels)
            # Calculate gradients
            loss.backward()
            # Change the model parameters based on gradients
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        # Check how well we're doing
        evaluate_wrapper(net, val_dataloader, train_dataloader, device, val_curve, train_curve)
    
    print('Finished Training')
    print(f'Training duration: {np.round(time.time()-start_t, 3)} seconds')
    return train_curve, val_curve