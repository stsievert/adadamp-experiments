
""""
TODO:

- [ ] write out proper train/test split. Do validation split in this script.

"""

from pathlib import Path
import sys
from typing import Optional, Dict, Any
import pickle
from copy import deepcopy
import json
import itertools
from time import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

import train
print(train.__file__)

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

def md5(x: Any) -> str:
    import hashlib
    result = hashlib.md5(str(x).encode("ascii"))
    return result.digest().hex()

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
        noisy=False,
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
        self.noisy = noisy

    def fit(self, X, y=None):

        seed = int(np.unique(np.asarray(X).flatten())[0])
        print(X, seed)
        keys = [
            "damper",
            "lr",
            "initial_batch_size",
            "max_batch_size",
            "momentum",
            "dampingdelay",
            "dampingfactor",
            "rho",
            "dwell",
            "weight_decay",
            "nesterov",
            "noisy",
        ]
        kwargs = {k: getattr(self, k) for k in keys}
        ident = md5(tuple(kwargs.items()))
        if kwargs["damper"] == "geodamplr":
            kwargs["damper"] = "geodamp"
            kwargs["max_batch_size"] = self.initial_batch_size

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

        out_dir = Path(__file__).absolute().parent / "data-tuning" / "runs"
        out_data = pd.DataFrame(self.data_)
        with open(out_dir / "{self.damper}-{ident}.pkl.zip", "wb") as f:
            out_data.to_pickle(f)
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
        "max_batch_size": [256, 512, 1024, 2048, 4096],#, 8192, 16384],
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

    def run(k, wd, dwell, momentum, lr, ibs, noisy):
        start = time()
        print(f"Fitting {k}th param w/ {dwell=}, {lr=}, {momentum=}, {wd=}")
        kwargs = {"dwell": dwell, "lr": lr, "momentum": momentum, "weight_decay": wd}
        m = Wrapper(
            MLP(), X_train, y_train, X_test, y_test,
            epochs=200, verbose=True, tuning=2,
        )
        m.set_params(
            damper="adadampnn",
            dwell=dwell,
            initial_batch_size=ibs,
            max_batch_size=4096,
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=wd,
            noisy=noisy,
        )
        seeds = 3 + (np.arange(3 * 2) // 2).reshape(3, 2)
        try:
            m.fit(seeds.astype(int))
        except Exception as e:
            print(f"Failed with {e} for job w/ {dwell=}, {lr=}, {momentum=}, {wd=}")
            import logging
            logging.exception(e)
            return

        print(f"    took {time() - start}s")
        out_f = DIR / "data-tuning" / "sweep" / f"d={dwell}-lr={lr}-m={momentum}-wd={wd}-ibs={ibs}-noisy={noisy}.pkl.zip"
        pd.DataFrame(m.data_).to_pickle(out_f)
        return

    # most important first
    noisy = [True, False]
    ibs = [2, 4, 8, 16, 16 * 2, 16 * 3, 16 * 4, 16 * 5]
    lrs = [1e-3, 0.1e-3, 0.3e-3]
    momentums = [0.8, 0.6, 0.7, 0.9, 0.95]
    dwells = [300, 100, 1000, 3000]
    wds = [1e-5, 1e-6, 1e-4]
    params = list(itertools.product(wds, dwells, momentums, lrs, ibs, noisy))
    print("len(params) =", len(params))

    from joblib import Parallel, delayed
    Parallel(n_jobs=64)(delayed(run)(k, *param) for k, param in enumerate(params))
    sys.exit(0)


    # adagrad uses <=426M
    n_params = 64
    for damper in [
        "geodamplr",
        "adadampnn",
        "geodamp",
        "sgd",
        "gd",
        "adagrad",
    ]:
        print(f"## Training w/ {damper}")
        space = deepcopy(base_search_space)
        space.update(damper_search_space.get(damper, {}))
        space.update({"damper": [damper]})

        est = MLP()
        epochs = 200 if damper != "gd" else 500
        m = Wrapper(
            est, X_train, y_train, X_test, y_test,
            epochs=epochs, verbose=False, tuning=1,
        )
        search = RandomizedSearchCV(
            m, space, n_iter=n_params, n_jobs=32, refit=True, verbose=3,
            random_state=42, cv=[([0], [1, 2]), ([1], [1, 2]), ([2], [0, 1])]
        )
        seeds = 1 + (np.arange(3 * 2) // 2).reshape(3, 2)
        search.fit(seeds.astype(int))

        with open(OUT / f"search-{damper}.pkl", "wb") as f:
            pickle.dump(search, f)
