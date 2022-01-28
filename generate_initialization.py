# Generate initialization for parameter for main_v2.py
import torch
from torch import nn

# The same neural network as main_v2.py
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.a_block = nn.Sequential(
            nn.Linear(10,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,10)
        )

    def forward(self, x):
        logits = self.flatten(x)
        logits = self.a_block(logits)
        return logits

if __name__ == "__main__":
    for seed in range(10):
        torch.manual_seed(seed)
        model = NeuralNetwork()
        filename = './initialization/v2_init_' + str(seed) + '.pth'
        torch.save(model.state_dict(), filename)