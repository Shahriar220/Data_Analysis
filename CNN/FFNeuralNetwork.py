#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:55:58 2024

@author: Shahriar
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


BATCH_SIZE = 120
EPOCHS = 10
LEARNING_RATE = .001


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_dataset():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss:{loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("=====================")
    print("Training is done")


if __name__ == "__main__":
    train_data, _ = download_mnist_dataset()
    print("Mnist data downloaded")

    # train data loader for the train set
    train_data_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE)

    # build model
    # FOR GPU You could use KUDA
    # FOR Goribs you should use CPU (mara khaiba ajibon ;p)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    feed_forward_net = FeedForwardNet().to(device)

    # Instantiate a loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    train(feed_forward_net, train_data_loader,
          loss_fn, optimizer, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model Trained Successfully")
