## import packages here
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

## enable CUDA acceleration here
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device.")

## define model here (structure of the neural network)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1,1)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.relu1(x)
        return logits


## generate training data here

## generate a model instance and send it to device here
print("Generating model...")
model = NeuralNetwork().to(device)
print(model)

## define loss function and optimizer
print("Generating loss function and optimizer ...")

## start training
print("start training ...")