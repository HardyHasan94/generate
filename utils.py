"""
Utils
-----
This file contain helper functions.
"""

import random

import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets
from torchvision import transforms

from gan import CustomDataset


def make_data(n, name='FACES'):
    """
    Make a dataset for the given problem.

    Parameters
    ----------
    n: int
        Number of datapoints to create.
    name: {'FACES', 'SINE', 'MNIST'}
        Problem name.

    Returns
    -------
    data: numpy.ndarray
        An array of the generated data.

    """
    if name == 'FACES':
        data = make_faces(n)
    elif name == 'SINE':
        data = make_sine(n)
    elif name == 'MNIST':
        data = make_mnist(n)

    return data


def set_seed(seed):
    """
    Seed random number generators.
    Parameters
    ----------
    seed: int
        A seed integer.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return None


def make_faces(n):
    """
    Create and return random face and non-face datapoints, together with their labels.

    Parameters
    ----------
    n: int
        Number of datapoints.

    Returns
    -------
    dataset: torch.utils.data.DataSet
        A custom DataSet object, that can be consumed by a dataloader.
    """
    data = np.empty((n, 4))
    main_diagonal = scipy.stats.norm(loc=0.1, scale=0.05)
    anti_diagonal = scipy.stats.norm(loc=0.95, scale=0.05)

    data[:, 0], data[:, 1], data[:, 2], data[:, 3] = anti_diagonal.rvs(size=n), main_diagonal.rvs(size=n), \
                                                     main_diagonal.rvs(size=n), anti_diagonal.rvs(size=n)
    data = data.clip(0, 1)

    data, labels = torch.tensor(data, dtype=torch.float32), torch.ones(n)
    dataset = CustomDataset(data, labels)

    return dataset


def make_sine(n):
    """
    Generate two-dimensional datapoints from a sine curve.
    A datapoint is (x1, np.sin(x1)) where x1 in [0, 2*np.pi]

    Parameters
    ----------
    n: int
        Number of points to generate.

    Returns
    -------
    dataset: torch.utils.data.DataSet
        A custom DataSet object, that can be consumed by a dataloader.

    """
    x1 = 2 * np.pi * np.random.random((n, 1))
    x2 = np.sin(x1)
    data = np.hstack((x1, x2))
    data, labels = torch.tensor(data, dtype=torch.float32), torch.ones(n)
    dataset = CustomDataset(data, labels)

    return dataset


def make_mnist(n):
    """
    Retrieves mnist images of handwritten digits.
    An image is a 28x28 tensor.

    Parameters
    ----------
    n: int
        Number of images to return.

    Returns
    -------
    data: torchvision.datasets.mnist.MNIST
        A TrainSet object returned by the torchvision.datasets.

    """
    # transform applies two transformation to an image which originally is in range [0, 255]
    # first, ToTensor makes a tensor of it, and normalizing it to [0, 1]. Normalize standardizes it
    # to the range [-1, 1].
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = torchvision.datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
    return train_set


def plot_faces(data, n, return_figure=False):
    """
    Plot the faces' data.

    Parameters
    ----------
    data: numpy.ndarray
        dataset
    n: int
        Number of datapoints to plot.
    return_figure: bool
        Whether to return a figure instead of showing it.

    Returns
    -------
    None or fig if return_figure is True

    """
    faces = data[:n, :].reshape((n, 2, 2))

    fig, axes = plt.subplots(nrows=5, ncols=n//5, figsize=(7, 7))

    for ax, face in zip(axes.flat, faces):
        ax.imshow(face, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    if return_figure:
        return fig
    else:
        plt.show()


def plot_sine(data, return_figure=False):
    """
    Make a scatter plot of the sine-curve with given data.

    Parameters
    ----------
    data: numpy.ndarray
        Dataset of points (x1, x2).
    return_figure: bool
        Whether to return a figure instead of showing it.

    Returns
    -------
    None or fig if return_figure is True
    """
    fig, ax = plt.subplots()
    ax.set(xlabel='x1', ylabel='x2', title='Sine-curve.')
    ax.scatter(data[:, 0], data[:, 1], color='navy')
    if return_figure:
        return fig
    else:
        plt.show()


def plot_mnist(data, n, return_figure=False):
    """
    Plot the faces' data.

    Parameters
    ----------
    data: torch.Tensor
        Tensor of images.
    n: int
        Number of datapoints to plot.
    return_figure: bool
        Whether to return a figure instead of showing it.

    Returns
    -------
    None or fig if return_figure is True

    """
    fig, axes = plt.subplots(5, n // 5, figsize=(10, 10))
    for i in range(n):
        ax = axes.flat[i]
        ax.imshow(data[i].reshape((28, 28)), cmap='gray_r')
        ax.set_xticks([])
        ax.set_yticks([])

    if return_figure:
        return fig
    else:
        plt.show()


def plot_data(data, name='FACES', n=-1, return_figure=False):
    """
    Plot a dataset for the given problem.

    Parameters
    ----------
    data: numpy.ndarray
        A dataset.
    name: {'FACES', 'SINE', 'MNIST'}
        Problem name.
    n: int
        Number of datapoints to plot.
    return_figure: bool
        Whether to return a figure instead of showing it.

    Returns
    -------
    None or fig if return_figure is True
    """
    if name == 'FACES':
        fig = plot_faces(data=data, n=n, return_figure=return_figure)
    elif name == 'SINE':
        fig = plot_sine(data=data, return_figure=return_figure)
    elif name == 'MNIST':
        fig = plot_mnist(data=data, n=n, return_figure=return_figure)

    return fig


def plot_loss(losses):
    """
    Make a plot of generator and discriminator losses.

    Parameters
    ----------
    losses: list of tuples
        List of tuples, where each tuple is (generator_loss, discriminator_loss,
        discriminator_real_preds_mean, discriminator_fake_preds_mean)

    Returns
    -------
    None

    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set(xlabel='Iteration', ylabel='Loss', title='Generator and Discriminator training loss.')
    # ax.plot(range(len(losses)), [el[0] for el in losses], ls='--', color='olive', label='Generator')
    # ax.plot(range(len(losses)), [el[0] for el in losses], ls='--', color='tomato', label='Discriminator')
    ax.plot(range(len(losses)), [el[0] for el in losses], ls='--', color='darkslateblue', label='D(x)')
    ax.plot(range(len(losses)), [el[1] for el in losses], ls='--', color='darkcyan', label='D(G(z))')
    plt.legend()
    plt.show()

    return None


def make_dataloader(data, batch_size):
    """
    Creates a and returns a dataloader for some dataset.
    Parameters
    ----------
    data: numpy.ndarray
        Dataset
    batch_size: int
        Batch size

    Returns
    -------
    dataloader: torch.utils.data.DataLoader
        A dataloader.

    """
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    return dataloader


def create_noise(shape):
    """
    Sample noise data from a standard normal distribution of given shape.

    Parameters
    ----------
    shape: tuple
        (n_rows, n_features)

    Returns
    -------
    noise: torch.Tensor
        Sampled noise tensor.

    """
    noise = torch.randn(shape, dtype=torch.float32)
    return noise

# =============== END OF FILE # ===============
