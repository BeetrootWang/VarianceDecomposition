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
# TODO: modify the neural network structure and dimensions
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

## define costomized loss here
def objective_function(y1,y2):
    # TODO: define loss function
    pass

## generate training data here
# TODO: generate dataloader
class my_dataset_object(Dataset):
    "my costomized dataset"
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

my_train_dataset, my_test_dataset = my_dataset_object()
train_dataloader = DataLoader(my_train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_dataloader = DataLoader(my_test_dataset, batch_size=4, shuffle=True, num_workers=0)

## generate a model instance and send it to device here
print("Generating model...")
model = NeuralNetwork().to(device)
print(model)

## define loss function and optimizer
print("Generating loss function and optimizer ...")
# TODO: modify loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

## define training main loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

## define the testing main loop (may not be necessary here
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # TODO: have a look at pred.argmax
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f} \n")



## start training
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1} \n ---------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Finished!")