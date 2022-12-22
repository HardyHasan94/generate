"""
Constants
-----
This file contain constants such as hyperparameters as well as network architecture definitions for each problem.
"""

import torch
import torch.nn as nn

ARCHITECTURES = {
    'FACES': {
        'Generator': {
            'in_features': 4,
            'out_features': (-1, 4),
            'model':  nn.Sequential(nn.Linear(in_features=4, out_features=4), nn.Sigmoid())
            },
        'Discriminator': {
            'in_features': 4,
            'out_features': 1,
            'model':  nn.Sequential(nn.Linear(in_features=4, out_features=1), nn.Sigmoid())
            },
    },

    'SINE': {
        'Generator': {
            'in_features': 2,
            'out_features': (-1, 2),
            'model':  nn.Sequential(
                nn.Linear(in_features=2, out_features=16),
                nn.ReLU(),
                nn.Linear(in_features=16, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=2),
            )
            },
        'Discriminator': {
            'in_features': 2,
            'out_features': 1,
            'model':  nn.Sequential(
                nn.Linear(in_features=2, out_features=256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=64, out_features=1),
                nn.Sigmoid()
            )
            },
    },

    'MNIST': {
        'Generator': {
            'in_features': 100,
            'out_features': (-1, 784),
            'model':  nn.Sequential(
                nn.Linear(in_features=100, out_features=256),
                nn.LeakyReLU(0.2),
                nn.Linear(in_features=256, out_features=512),
                nn.LeakyReLU(0.2),
                nn.Linear(in_features=512, out_features=1024),
                nn.LeakyReLU(0.2),
                nn.Linear(in_features=1024, out_features=784),
                nn.Tanh()
            )
            },
        'Discriminator': {
            'in_features': 784,
            'out_features': 1,
            'model':  nn.Sequential(
                nn.Linear(in_features=784, out_features=1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(in_features=1024, out_features=512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(in_features=512, out_features=256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(in_features=256, out_features=1),
                nn.Sigmoid()
            )
            },
    }
}


PARAMETERS = {
    'FACES': {
        'lr': 0.001,
        'epochs': 70,
        'batch_size': 32,
        'seed': 2022,
        'n_training_samples': 2**11,
        'n_eval_samples': 40
    },

    'SINE': {
        'lr': 0.001,
        'epochs': 50,
        'batch_size': 32,
        'seed': 2022,
        'n_training_samples': 2**13,
        'n_eval_samples': 200,
    },

    'MNIST': {
        'lr': 0.0001,
        'epochs': 30,
        'batch_size': 32,
        'seed': 2022,
        'n_training_samples': 60_00,
        'n_eval_samples': 40,
    }
}


# =============== END OF FILE # ===============
