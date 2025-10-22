#!/usr/bin/env python3

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the device to use for training
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # Flatten the input tensor
        # Define a sequential model with linear layers and ReLU activations
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # Input 28x28, lin trans to 512 features (y = Wx + b)
            nn.ReLU(), # activation function (f = max(0, x))
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), # output 10 classes
        )

    def forward(self, x):
        # defines how input data is propagated through the network
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    # X = torch.rand(1, 28, 28, device=device)
    # print(model)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # print(f"Logits: {logits}")
    # print(f"Predicted probabilities: {pred_probab}")
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")

    print("----------\nModel Layers")
    input_image = torch.rand(3,28,28)
    print(f"input_image = {input_image.size()}")
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(f"flat_image = {flat_image.size()}")
    layer1 = nn.Linear(in_features=28*28, out_features=20)
    hidden1 = layer1(flat_image)
    print(f"hidden1 = {hidden1.size()}")
    print(f"input_image = {input_image}\n\n")
    print(f"flat_image = {flat_image}\n\n")
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")

    # USe sequantial to define a neural network model
    seq_modules = nn.Sequential(
        flatten,
        layer1, # linear 28*28 to 20
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)
    print(f"Logits from sequential model: {logits.size()}")
    print(f"Logits: {logits}")
    softmax = nn.Softmax(dim=1) # sum of probs in dim=1 (columns) is 1
    pred_probab = softmax(logits)
    print(f"Predicted probabilities: {pred_probab}")
    print(f"sum of probabilities: {pred_probab.sum(dim=1)}")
    print(f"Predicted class: {pred_probab.argmax(1)}")

    # MOdel structure
    print(f"-----------\nModel structure: {model}\n\n")
    print("Model parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")