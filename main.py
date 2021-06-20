import random

import torch as t
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from network import AutoEncoder
from torch import nn
import time
import utils

batch_size = 64
learning_rate = 1e-3
epochs = 15
mnist_size = 28*28

start_time = time.time()

training_data = datasets.MNIST(root="./data",
                            train=True,
                            download=True,
                            transform=ToTensor())

test_data = datasets.MNIST(root="./data",
                        train=False,
                        download=True,
                        transform=ToTensor())

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = 'cuda' if t.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = AutoEncoder(input_shape=mnist_size).to(device)
loss_fn = nn.MSELoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        target = np.reshape(X, (X.shape[0], mnist_size))
        pred = model(X).to(device)
        loss = loss_fn(pred, target).to(device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss = 0

    with t.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            batch_samples_size = X.shape[0]
            random_sample = random.randrange(0, batch_samples_size)
            target = np.reshape(X, (batch_samples_size, mnist_size))
            pred = model(X).to(device)
            source_img = np.reshape(X[random_sample], (28, 28))
            pred_img = np.reshape(pred[random_sample], (28, 28))
            utils.show_tensor(source_img)
            utils.show_tensor(pred_img)

            test_loss += loss_fn(pred, target).item()

    test_loss /= size
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")
print("--- %s seconds ---" % (time.time() - start_time))