"""
Generative adversarial networks
-----
This file contain the class definitions of both the generator and discriminator networks.
"""

import random
from time import perf_counter
from datetime import timedelta

import numpy as np
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets
from torchvision import transforms


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.params = params
        self.in_features = params['architecture']['Discriminator']['in_features']
        self.out_features = params['architecture']['Discriminator']['out_features']
        self.model = params['architecture']['Discriminator']['model']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params['lr'])

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Input datapoint(s) tensor

        Returns
        -------
        pred: torch.tensor
            prediction of discriminator
        """
        x = x.view(x.size(0), self.in_features)
        pred = self.model(x)
        return pred

    def predict(self, x):
        """
        Make prediction(s) on datapoint(s).

        Parameters
        ----------
        x: torch.tensor
            Input data points of shape (n, ...)

        Returns
        -------
        pred: torch.tensor
            Prediction probabilities.
        """
        pred = self.forward(x)
        return pred

    def optimize(self, real, fake):
        """
        Perform an optimization step, which is using stochastic gradient ascent on the loss
        mean(y_label*ln(D(real)) + (1-y_label)*ln(1-D(fake))).

        Parameters
        ----------
        real: torch.Tensor
            Minibatch of real data.

        fake: torch.Tensor
            Minibatch of generated fake data.

        Returns
        -------
        loss: float
            Error cost of the predictions.
        mean_pred_real: float
            mean value of the predictions for the real part.
        mean_pred_fake: float
            mean value of the predictions for the fake part.
        """
        loss_func = nn.BCELoss()
        self.optimizer.zero_grad()

        # real part update
        real_preds = self.predict(real)
        real_labels = torch.ones_like(real_preds)
        loss_real = loss_func(real_preds, real_labels)
        loss_real.backward()

        # fake part update
        fake_preds = self.predict(fake)
        fake_labels = torch.zeros_like(fake_preds)
        loss_fake = loss_func(fake_preds, fake_labels)
        loss_fake.backward()

        self.optimizer.step()

        d_loss = round(loss_real.item() + loss_fake.item(), 2)

        return d_loss, round(real_preds.mean().detach().item(), 2), round(fake_preds.mean().detach().item(), 2)


class Generator(nn.Module):

    def __init__(self, params):
        super(Generator, self).__init__()
        self.params = params
        self.in_features = params['architecture']['Generator']['in_features']
        self.out_features = params['architecture']['Generator']['out_features']
        self.model = params['architecture']['Generator']['model']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params['lr'])

    def forward(self, noise):
        """
        Forward pass.

        Parameters
        ----------
        noise: torch.Tensor
            A minibatch of random data sampled from a standard normal distribution.

        Returns
        -------
        gen: torch.tensor
            Tensor of generated data.
        """
        gen = self.model(noise)
        return gen

    def generate(self, noise):
        """
        Generate datapoints.
        Parameters
        ----------
        noise: torch.Tensor
            A minibatch of random data sampled from a standard normal distribution.

        Returns
        -------
        data: torch.tensor
            tensor of the generated datapoints
        """
        data = self.forward(noise)
        return data

    def optimize(self, predicted_probs):
        """
        Perform an optimization step, which is using stochastic gradient ascent on the loss
        mean(ln(D(fake))).

        Parameters
        ----------
        predicted_probs: torch.Tensor
            Predicted probabilities by the discriminator for a batch of generated data by the generator.

        Returns
        -------
        loss: float
            Error cost of the predictions.
        """
        self.optimizer.zero_grad()
        loss_func = nn.BCELoss()
        loss = loss_func(predicted_probs, torch.ones_like(predicted_probs))
        loss.backward()
        self.optimizer.step()

        loss = round(loss.item(), 2)
        return loss


class CustomDataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, data, labels):
        """
        Initialize a custom DataSet.
        Parameters
        ----------
        data: torch.Tensor
            The datapoints.
        labels: torch.Tensor
            The labels.
        """
        super(CustomDataset, self).__init__()
        self.labels = labels
        self.data = data

    def __len__(self):
        """
        Denotes the total number of samples

        Returns
        -------
        length: int
            Length of all datapoints
        """
        length = len(self.data)
        return length

    def __getitem__(self, index):
        """
        Generates one sample of data.

        Parameters
        ----------
        index: int

        Returns
        -------
        X: torch.Tensor
            Corresponding datapoint at the index.
        label: torch.Tensor
            Corresponding label.

        """
        # Select sample
        X = self.data[index]
        y = self.labels[index]

        return X, y


# =============== END OF FILE # ===============
