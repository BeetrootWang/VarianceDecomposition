## Description
# This file run simple experiment with known (pre-saved)
#   - dataset
#   - parameter initialization
# Used for computing sample variance of the forward function at a specific input x
# (and other relevant sample statistics)

## import packages here
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

## define model here (structure of the neural network)
# simplest version: f(x) = ax + b; one layer no activation
# TODO: define a more complicated problem (Neural Network)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(1,1)

    def forward(self, x):
        logits = self.linear1(x)
        return logits

## generate training data here
# TODO:  randomness into the dataset
class my_dataset_object(Dataset):
    "my costomized dataset"
    def __init__(self, datapoints_x):
        self.datapoints_x = torch.from_numpy(datapoints_x)
        # underlying f^* is identity
        self.datapoints_y = self.datapoints_x

    def __len__(self):
        return len(self.datapoints_x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.datapoints_x[idx]
        y = self.datapoints_y[idx]
        xy_pairs = {"x": x, "y": y}
        return xy_pairs

## define training main loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, xy_pairs in enumerate(dataloader):
        X = xy_pairs["x"]
        y = xy_pairs["y"]
        X,y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%1 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

## define the testing main loop (may not be necessary here
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for xy_pairs in dataloader:
            X = xy_pairs["x"]
            y = xy_pairs["y"]
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def main_ijk(ii,jj,kk):
    ## get the kk-th training result for ii-th dataset, jj-th initialization
    ## parameters
    train_dataset_size = 3
    test_dataset_size = 100
    batch_size = 1
    epochs = 500

    learning_rate = 1e-3

    ## generate dataloader
    training_filename = 'data/training_1000_' + str(ii) + '.npy'
    testing_filename = 'data/testing_1000_' + str(ii) + '.npy'
    my_train_dataset = my_dataset_object(np.load(training_filename))
    my_test_dataset = my_dataset_object(np.load(testing_filename))
    import pdb; pdb.set_trace()
    train_dataloader = DataLoader(my_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(my_test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## generate a model instance and send it to device here
    print("Generating model...")
    model = NeuralNetwork().to(device)
    # Load initialization (in order to eliminate the variance induced by random initialization)
    # Recall: Q_{PV} consists of 1. initialization 2. data ordering
    model.load_state_dict(torch.load("simple_model_fixed_init_v1.pth"))
    print(model)

    ## define loss function and optimizer
    print("Generating loss function and optimizer ...")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ## start training
    for t in range(epochs):
        print(f"Epoch {t+1} \n ---------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Finished!")

    ## save models
    torch.save(model.state_dict(), "model.pth")
    print("saved PyTorch Model State to model.pth")
    print(f"\\hat a = {model.state_dict()['linear1.weight'].item():>7f} , \\hat b = {model.state_dict()['linear1.bias'].item():>7f}")

## main function
if __name__ == "__main__":

    ## enable CUDA acceleration here
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device.")

    ## run a single training procedure
    main_ijk(1,1,1)
