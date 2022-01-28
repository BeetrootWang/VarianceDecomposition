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
import pandas as pd

## define model here (structure of the neural network)
# simplest version: f(x) = ax + b; one layer no activation
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

## generate training data here
class my_dataset_object(Dataset):
    "my costomized dataset"
    def __init__(self, datapoints_x):
        self.datapoints_x = torch.from_numpy(datapoints_x)
        self.datapoints_x = self.datapoints_x.float() #  Add this to meet the requirement of the input for nn.Linear()

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

        if (batch+1)%200 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
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
    batch_size = 1
    epochs = 1

    learning_rate = 1e-3

    ## generate dataloader
    training_filename = 'data/training_1000_' + str(ii) + '.npy'
    testing_filename = 'data/testing_1000_' + str(ii) + '.npy'
    my_train_dataset = my_dataset_object(np.load(training_filename))
    my_test_dataset = my_dataset_object(np.load(testing_filename))
    train_dataloader = DataLoader(my_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(my_test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## generate a model instance and send it to device here
    print("Generating model...")
    model = NeuralNetwork().to(device)
    # Load initialization (in order to eliminate the variance induced by random initialization)
    # Recall: Q_{PV} consists of 1. initialization 2. data ordering
    model.load_state_dict(torch.load('./initialization/v2_init_' + str(jj) + '.pth'))
    print(model)

    # kk is the random seed for the remaining operations (data-ordering)
    torch.manual_seed(kk)

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

    return model.state_dict()
    # ## save models
    # torch.save(model.state_dict(), "model.pth")
    # print("saved PyTorch Model State to model.pth")

## main function
if __name__ == "__main__":

    ## enable CUDA acceleration here
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device.")

    ## run iid results with n1 = n2 = n3 = n = 10
    n = 10
    for ii in range(n):
        for jj in range(n):
            for kk in range(n):
                model_state_dict = main_ijk(ii,jj,kk)
                w = torch.tensor([]).to(device)
                for key in model_state_dict.keys():
                    w = torch.cat((w, model_state_dict[key].reshape(-1)))
                w = pd.DataFrame(w.reshape(-1).cpu()).T
                w.index = [(ii,jj,kk)]
                if ii==0 and jj==0 and kk==0:
                    result = w
                else:
                    result = pd.concat([result,w])
                if kk == n-1:
                    result.to_csv('tmp_saved_data.csv')

    ## save result

