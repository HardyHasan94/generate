"""
Main
-----
This file contain the main training loop of the gan system.
"""

import argparse

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

import wandb

from constants import ARCHITECTURES, PARAMETERS
from utils import set_seed, make_data, make_dataloader, make_sine, make_mnist, \
    make_faces, plot_mnist, plot_data, plot_loss, plot_sine, plot_faces, create_noise
from gan import Generator, Discriminator


parser = argparse.ArgumentParser(description="Train a GAN system on a problem among three problems.")
parser.add_argument('--seed', required=True, type=int, default=2022, help="Select integer value seed for reproduction.")
parser.add_argument('--problem', required=True, type=str, default='FACES', choices=['FACES', 'SINE', 'MNIST'],
                    help="Select a problem to run gan on.")
parser.add_argument('--wandb', required=False, action='store_true', default=False,
                    help="Whether to log training progress to wandb. If True, "
                         "then user must be logged in to wandb.")


def train(params, data, log_wandb):
    """
    Perform the training of the Generative and Adversarial networks

    Parameters
    ----------
    params: dict
        Dictionary of hyperparameters

    data: numpy.ndarray
        The training dataset.

    log_wandb: bool
        Whether to watch model on wandb.

    Returns
    -------
    generator: object
        The generator network class.

    """
    # start background wandb process
    generator = Generator(params=params)
    discriminator = Discriminator(params=params)

    if log_wandb:
        # TODO: define the following parameters to use logging in wandb correctly, and uncomment the next two lines.
        # run = wandb.init(name=PROBLEM_NAME, project=WANDB_PROJECT_NAME, entity=USER_NAME, config=PARAMETERS_DICTIONARY)
        # wandb.watch(models=[discriminator, generator], log='all', log_freq=1, log_graph=True)
        raise NotImplementedError("Some parameters need to be defined.")

    dataloader = make_dataloader(data=data, batch_size=params['batch_size'])
    losses = []
    batch_size = params['batch_size']
    n_iterations = params['epochs']*(params['n_training_samples']/params['batch_size'])
    discriminator_input_dim = params['architecture']['Discriminator']['in_features']
    generator_input_dim = params['architecture']['Generator']['in_features']
    n_eval_samples = params['n_eval_samples']
    problem_name = params['problem_name']
    evaluation_noise = create_noise((n_eval_samples, generator_input_dim))
    iteration = 0

    for epoch in range(params['epochs']):
        for count, (real_data, _) in enumerate(dataloader):
            real_data = real_data.reshape((-1, discriminator_input_dim))  # reshape image to vector
            n_rows = real_data.shape[0]

            # --------------- TRAIN THE DISCRIMINATOR ---------------
            # sample a minibatch of real and a minibatch of fake data.
            # --------------------------------------------------------

            noise = create_noise((n_rows, generator_input_dim))
            fake_data = generator.generate(noise).detach()
            loss_discriminator, d_real_preds_mean, d_fake_preds_mean = discriminator.optimize(real_data, fake_data)
            losses.append((d_real_preds_mean, d_fake_preds_mean))

            # --------------- TRAIN THE GENERATOR ---------------

            noise = create_noise((n_rows, generator_input_dim))
            fake_data = generator.generate(noise)
            generated_probs = discriminator.predict(fake_data)
            loss_generator = generator.optimize(generated_probs)

            # --------------- LOGGING DATA ---------------

            if iteration % 500 == 0:
                print(f"iteration: {iteration} loss_discriminator: {loss_discriminator} "
                      f"real_preds_mean: {d_real_preds_mean} fake_preds_mean: {d_fake_preds_mean}")
            if log_wandb:
                wandb.log({'d_real_probs_mean': d_real_preds_mean,
                           'd_fake_probs_mean': d_fake_preds_mean,
                           'loss_discriminator': loss_discriminator,
                           'loss_generator': loss_generator},
                          step=iteration)
            iteration += 1

        if log_wandb:
            generated_data = generator.generate(evaluation_noise).detach().numpy()
            fig = plot_data(data=generated_data, name=problem_name, n=n_eval_samples, return_figure=True)
            wandb.log({'images': [wandb.Image(fig, caption="Training progress for Generator network.")]})
            plt.close(fig)

    if log_wandb is False:
        plot_loss(losses)

    return generator


if __name__ == "__main__":
    args = parser.parse_args()
    problem_name, seed, log_wandb = args.problem, args.seed, args.wandb

    params = PARAMETERS[problem_name]
    set_seed(seed)
    params['architecture'] = ARCHITECTURES[problem_name]
    params['problem_name'] = problem_name
    n_training_samples = params['n_training_samples']
    n_eval_samples = params['n_eval_samples']
    generator_input_dim = params['architecture']['Generator']['in_features']

    data = make_data(n=n_training_samples, name=problem_name)

    # training phase
    start_time = perf_counter()
    generator = train(params, data, log_wandb)
    end_time = perf_counter()
    print(f"Training time: _ _ _ _ _{str(timedelta(seconds=end_time-start_time))}_ _ _ _ _")


# =============== END OF FILE # ===============
