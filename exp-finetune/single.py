
""""
TODO:

- [ ] write out proper train/test split. Do validation split in this script.

"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any
import pickle
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

import train

DIR = Path(__file__).absolute().parent
DS_DIR = DIR / "dataset"
OUT = DIR / "data-tuning"
sys.path.extend([str(DIR.parent), str(DIR.parent / "train")])

class MLP(nn.Module):
    def __init__(self, in_feats=768, layer1=400, layer2=200, n_labels=70):
        super().__init__()
        self.map1 = nn.Linear(in_feats, layer1)
        self.map2 = nn.Linear(layer1, layer2)
        self.map3 = nn.Linear(layer2, n_labels)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        x = self.map3(x)
        y_hat = F.log_softmax(x, dim=1)  # from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#log_softmax , v2.5
        return y_hat

def loguniform_m1(a, b, n=10_000, random_state=42):
    a, b = min(a, b), max(a, b)
    x = loguniform(a, b).rvs(size=n, random_state=random_state)
    return (1 - x).tolist()

def ints(a, b):
    return list(range(a, b + 1))

class Wrapper(BaseEstimator):
    def __init__(self,
        model,
        X_train, y_train, X_test, y_test,
        tuning: int = 1,
        epochs=100,
        verbose=1,
        seed=42,
        *,
        lr=1e-2,
        initial_batch_size=32,
        max_batch_size=256,
        momentum=0.99,
        damper="adagrad",
        dampingdelay=20,
        dampingfactor=5,
        rho=0.99,
        dwell=5,
        weight_decay=0.005,
        nesterov=True,
    ):
        self.model = model
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.verbose = verbose
        self.epochs = epochs
        self.tuning = tuning
        self.seed = seed

        self.damper = damper
        self.lr = lr
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.momentum = momentum
        self.dampingdelay = dampingdelay
        self.dampingfactor = dampingfactor
        self.rho = rho
        self.dwell = dwell
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def fit(self, X, y=None):
        seed = int(np.unique(X.flatten())[0])
        print(X, seed)
        keys = [
            "damper",
            "lr",
            "initial_batch_size",
            "max_batch_size",
            "momentum",
            "damper",
            "dampingdelay",
            "dampingfactor",
            "rho",
            "dwell",
            "weight_decay",
            "nesterov",
        ]
        kwargs = {k: getattr(self, k) for k in keys}
        data, train_data, model, test_set = train.main(
            dataset=None,
            model=self.model,
            epochs=self.epochs,
            train_data=(self.X_train, self.y_train),
            test_data=(self.X_test, self.y_test),
            tuning=self.tuning + seed,
            test_freq=1,
            cuda=True,
            random_state=self.seed,
            init_seed=self.seed,
            verbose=self.verbose,
            **kwargs,
        )
        self.data_ = data
        self.model_ = model
        self.train_data_ = train_data
        return self

    def score(self, *args, **kwargs):
        return -1 * self.data_[-1]["test_loss"]

if __name__ == "__main__":
    with open(DS_DIR / "embedding.pt", "rb") as f:
        data = torch.load(f, weights_only=True)
    X_train, X_test, y_train, y_test = [data[k] for k in ["X_train", "X_test", "y_train", "y_test"]]

    assert tuple(X_train.shape) == (6667, 768) and tuple(X_test.shape) == (3333, 768)
    assert tuple(y_train.shape) == (6667, ) and tuple(y_test.shape) == (3333, )
    assert len(np.unique(y_train)) == 70

    est = MLP()
    params = {k: x for k, x in est.named_parameters()}
    nele = {k: x.nelement() for k, x in params.items()}
    assert sum(nele.values()) == 401_870
    
    base_search_space = {
        "lr": loguniform(1e-5, 1e-1),
        "initial_batch_size": [16, 32, 64, 96, 128, 160, 192, 224, 256],
        "max_batch_size": [None]*3 + [256, 512, 1024, 2048, 4096, 8192, 16384],
        "dwell": [1, 2, 5, 10, 20, 50],
        "weight_decay": loguniform(1e-8, 1e-4),
        "momentum": loguniform_m1(1e-1, 1e-3),
        "nesterov": [True],
    }
    damper_search_space: Dict[str, Dict[str, Any]] = {
        "adadamp": {},
        "geodamp": {"dampingdelay": ints(1, 10),
                    "dampingfactor": ints(1, 21)},
        "radadamp": {"rho": loguniform_m1(1e-5, 1e-1)},
        "adagrad": {"lr": loguniform(1e-3, 1e-1)},
    }

    # gd use {1384, 786, 1076}M
    epochs = 100
    n_params = 200

    for damper in [
        "sgd",
        #"radadamp",  # 1391M
        "adadamp",
        "gd",  # GPU mem 1378M, 90/300W.
        "adagrad",
        "geodamp",
        "adadelta",
    ]:
        print(f"## Training w/ {damper}")
        space = deepcopy(base_search_space)
        space.update(damper_search_space.get(damper, {}))
        space.update({"damper": [damper]})

        est = MLP()
        m = Wrapper(
            est, X_train, y_train, X_test, y_test,
            epochs=epochs, verbose=False, tuning=1,
        )
        search = RandomizedSearchCV(
            m, space, n_iter=n_params, n_jobs=-1, refit=False, verbose=3,
            random_state=42, cv=[([0], [1, 2]), ([1], [1, 2]), ([2], [0, 1])]
        )
        seeds = 1 + (np.arange(3 * 2) // 2).reshape(3, 2)
        search.fit(seeds.astype(int))

        with open(OUT / f"search-{damper}.pkl", "wb") as f:
            pickle.dump(search, f)
