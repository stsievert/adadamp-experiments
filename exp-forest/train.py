import logging
import math
import random
import os
from collections import namedtuple
from hashlib import sha256
from pprint import pprint
from time import perf_counter
from typing import TypeVar, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from distributed.protocol import serialize, deserialize
from sklearn.base import BaseEstimator
from skorch import NeuralNet
from skorch.classifier import NeuralNetClassifier


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 54
        n_layers = 6

        p = 1 / 3
        first_layer = [nn.Linear(54, n), nn.ReLU(), nn.Dropout(p=p)]
        layers = [
            [nn.Linear(n, n), nn.ReLU(), nn.Dropout(p=p)] for _ in range(n_layers)
        ]
        last_layers = [
            nn.Linear(n, 4 * n),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(4 * n, 8),
        ]

        all_layers = first_layer + sum(layers, []) + last_layers
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.layers(x)


class CovClassifier(BaseEstimator):
    def __init__(
        self,
        weight_decay=1e-4,
        lr=0.01,
        seed=42,
        device="cpu",
        tuning=False,
        momentum=0.9,
        opt="sgd",
    ):
        self.seed = seed
        self.tuning = tuning
        self.weight_decay = weight_decay
        self.lr = lr
        self.device = device
        self.momentum = momentum
        self.opt = opt

        kwargs = dict(
            optimizer=optim.SGD,
            optimizer__weight_decay=self.weight_decay,
            optimizer__momentum=self.momentum,
            optimizer__nesterov=True,
        )
        if opt == "adagrad":
            kwargs = dict(
                optimizer=optim.Adagrad,
                optimizer__weight_decay=self.weight_decay,
            )
        self.model = NeuralNet(
            module=Net,
            lr=lr,
            criterion=nn.CrossEntropyLoss,
            warm_start=True,
            max_epochs=1,
            batch_size=-1,
            train_split=None,
            device=device,
            **kwargs,
        )
        super().__init__()

    def _set_seed(self):
        seed = self.seed
        assert seed is not None, "Specify seed, don't leave seed=None"
        s = str(seed) * 10
        sha = sha256(bytes(s, "ascii"))
        randint = int("0x" + sha.hexdigest(), 0)
        capped = randint % (2 ** 32 - 1)

        torch.manual_seed(capped)
        random.seed(capped)
        return np.random.RandomState(capped)

    def initialize(self):
        self.rng_ = self._set_seed()
        self.model.initialize()
        #         self.model_ = Net()
        #         self.optimizer_ = optim.AdaGrad(weight_decay=self.weight_decay)
        assert self.model.initialized_ == True
        self.initialized_ = True

        self.history_ = []
        self.models_ = []
        self.meta_ = {
            "model_updates": 0,
            "num_examples": 0,
            "len_dataset": 25e3,  # not far from to kaggle's [1]
        }
        # [1]:https://www.kaggle.com/c/forest-cover-type-prediction/data
        if self.tuning:
            self.meta_["len_dataset"] *= 0.8
        return True
