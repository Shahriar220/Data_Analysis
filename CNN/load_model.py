#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:36:24 2024

@author: Shahriar
"""

import torch
import FFNeuralNetwork

from FFNeuralNetwork import download_mnist_dataset, FeedForwardNet

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":

    # load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # get validation data
    _, validation_data = download_mnist_dataset()

    input, target = validation_data[0][0], validation_data[0][1]

    # make an inference
    predicted, expected = predict(
        feed_forward_net, input, target, class_mapping)

    print(f"Predicted: '{predicted}',expected: '{expected}'")
