import datetime
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

import models.models as models


def time_left(t_start, n_iters, i_iter):
    iters_left = n_iters - i_iter
    time_per_iter = (datetime.datetime.now() - datetime.datetime.fromtimestamp(t_start)) / i_iter
    time_left = time_per_iter * iters_left
    return time_left


def plot_grad_flow(named_parameters):
    """
    source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and (p.grad is not None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.figure(figsize=(20, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


def model_mapping(name, input_dim, dropout):
    """
    Function to select the corresponding model to an input string.
    Used for fast switching between models
    """
    if name == 'ConvModelGender':
        return models.ConvModel(['gender'], input_dim, dropout)
    elif name == 'ConvModelAE':
        return models.ConvModel(['autoencoder'], input_dim, dropout)
    elif name == 'ConvModelGenderAE':
        return models.ConvModel(['gender', 'autoencoder'], input_dim, dropout)
    elif name == 'GenderClassifier':
        return models.GenderClassifier(input_dim)
    elif name == 'MayoNetGender':
        return models.MayoNet()
    elif name == 'MayoResNetGender':
        return models.MayoResNet()
    else:
        raise Exception("Unknown model: '{}'".format(name))
