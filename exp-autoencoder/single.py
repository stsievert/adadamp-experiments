
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
from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV

import train
print(train.__file__)

DIR = Path(__file__).absolute().parent
DS_DIR = DIR / "dataset"
OUT = DIR / "data-tuning2"
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
        epochs=200,
        verbose=1,
        seed=5968,
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
        reduction="mean",
        wait: int = 10,
        growth_rate=1e-3,
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
        self.reduction = reduction
        self.wait = wait
        self.growth_rate = growth_rate

    def fit(self, X, y=None):
        params = {
            k: v
            for k, v in self.get_params().items()
            if k not in ['X_test', 'X_train', 'model', 'seed', 'verbose', 'y_test', 'y_train', 'tuning', 'write']
        }
        params2 = [(k, params[k]) for k in sorted(list(params.keys()))]
        #ident = md5(tuple(clean(params2)))
        seed = 1 + int(np.unique(np.asarray(X).flatten())[0])
        ident = "-".join(
            f"{k[:3]}={v:0.6e}" if isinstance(v, float) else f"{k[:3]}={v}"
            for k, v in clean(params2)
        )
        #ident = f"{params['damper']}-tune"
        fname = f"{ident}-{seed}.pkl.zip"

        kwargs = params
        if "damplr" in kwargs["damper"] or "dampnnlr" in kwargs["damper"]:
            kwargs["damper"] = kwargs["damper"][:-2]
            kwargs["max_batch_size"] = self.initial_batch_size

        out_dir = OUT / "dampersweep"
        #if out_dir / fname in out_dir.iterdir():
        #    print(f"Skipping {fname}, already in directory", flush=True)
        #    self.df_ = pd.read_pickle(out_dir / fname)
        #    return self

        start = time()
        try:
            data, train_data, model, test_set = train.main(
                dataset="autoencoder",
                #model=self.model,
                #train_data=(self.X_train, self.y_train),
                #test_data=(self.X_test, self.y_test),
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
        except Exception as e:
            import logging
            logging.exception(e)
            return self
        print(f"    took {(time() - start) / 60:0.1f}min", flush=True)
        pd.DataFrame(data).to_pickle(out_dir / fname)
        df = pd.DataFrame(data)
        df.to_pickle(out_dir / fname)
        self.df_ = df
        return self

    def score(self, *args, **kwargs):
        return -1 * self.df_.iloc[-1]["test_loss"]

def clean(x):
    if isinstance(x, list):
        return [clean(i) for i in x]
    if isinstance(x, tuple):
        return tuple([clean(i) for i in x])
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (np.int32, np.int64, np.float64, np.float32)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def single_run():
    with open(DS_DIR / "embedding.pt", "rb") as f:
        data = torch.load(f, weights_only=True)
    X_train, X_test, y_train, y_test = [data[k] for k in ["X_train", "X_test", "y_train", "y_test"]]
    space = {
        "reduction": ["mean", "min", "median", "max"],
        "dwell": [1, 3, 10, 30, 100],
        "lr": loguniform(0.1e-3, 10e-3),
        "momentum": uniform(0.05, 0.95),
        "wait": [10, 30, 100, 300, 1000],
        "rho": uniform(0,0.95),
        "max_batch_size": [128, 256, 512, 1024, 2048, 4096],
        "initial_batch_size": [8, 16, 32, 64, 128, 512],
        "weight_decay": loguniform(1e-7, 1e-5),
    }
    base = Wrapper(
        MLP(), X_train, y_train, X_test, y_test,
        epochs=500, verbose=False, tuning=2,
        damper="padadamp",
        #damper="adadampnn",
    )
    base.set_params(
        lr=1e-3,
        weight_decay=1e-6,
        dwell=100,
        growth_rate=0.01,
        initial_batch_size=128,
        max_batch_size=1024,
        #wait=500,
        #rho=0.1,
        #reduction="min",
        momentum=0.99,
    )
    from sklearn.base import clone
    for damper in ["padadamp", "padadamplr"]:#, "adadampnn", "adadampnnlr"]:
        seeds = 1 + (np.arange(3 * 2) // 2).reshape(3, 2)
        to_fit = clone(base)
        print(to_fit.growth_rate)
        to_fit.set_params(damper=damper)
        to_fit.fit(seeds.astype(int))

    #for i in range(100):
    #    search = RandomizedSearchCV(
    #        base, space, n_iter=80, n_jobs=100, refit=False,
    #        random_state=42 + i,
    #        cv=[([0], [1, 2]), ([1], [1, 2]), ([2], [0, 1])],
    #    )
    #    search.fit(seeds.astype(int))
    #    print(f"\n\nsaving search {i}")
    #    with open(OUT / "searches" / f"rs-{i}.pkl", "wb") as f:
    #        pickle.dump(search, f)

if __name__ == "__main__":
    with open(DS_DIR / "embedding.pt", "rb") as f:
        data = torch.load(f, weights_only=True)
    X_train, X_test, y_train, y_test = [data[k] for k in ["X_train", "X_test", "y_train", "y_test"]]

    assert tuple(X_train.shape) == (6667, 768) and tuple(X_test.shape) == (3333, 768)
    assert tuple(y_train.shape) == (6667, ) and tuple(y_test.shape) == (3333, )
    assert len(np.unique(y_train)) == 70

    #est = MLP()
    #params = {k: x for k, x in est.named_parameters()}
    #nele = {k: x.nelement() for k, x in params.items()}
    #assert sum(nele.values()) == 401_870
    
    base_search_space = {
        "lr": loguniform(1e-5, 1e-1),
        "initial_batch_size": [2**i for i in range(4, 9 + 1)],
        "max_batch_size": [2**i for i in range(7, 13 + 1)],
        #"dwell": [1, 2, 5, 10, 20, 50],
        "dwell": loguniform(1, 100),
        "weight_decay": loguniform(1e-7, 1e-4),
        "momentum": loguniform_m1(1e-1, 1e-3),
        "nesterov": [True],
    }
    damper_search_space: Dict[str, Dict[str, Any]] = {
        "adadamp": {},
        "geodamp": {"dampingdelay": ints(1, 10),
                    "dampingfactor": ints(1, 21)},
        "radadamp": {"rho": loguniform_m1(1e-5, 1e-1)},
        "adagrad": {"lr": loguniform(1e-3, 1e-1)},
        "adadampnn": {

        },
    }

    padadamp_space = {
        #"lr": loguniform(1e-4, 1e-2),
        #"momentum": loguniform_m1(1e-1, 5e-3),
        #"weight_decay": loguniform(1e-7, 1e-5),
        "lr": loguniform(0.5e-4, 0.5e-1),
        "growth_rate": loguniform(1e-3, 1e2),
        "momentum": loguniform_m1(5e-1, 1e-3),
    }

    # adagrad uses <=426M
    #n_params = 15
    n_params = 125
    for i in range(20, n_params):
        for damper in [
            "adadampnn",
            "padadamp",
            "padadamplr",
            #"geodamplr",
            #"geodamp",
            #"adagrad",
            #"gd",
            #"sgd",
        ]:
            print(f"## Training w/ {damper}")
            space = deepcopy(base_search_space)
            space.update(damper_search_space.get(damper, {}))
            if "padadamp" in damper:
                space.update(padadamp_space)
            space.update({"damper": [damper]})

            epochs = 100 if damper != "gd" else 1000

            m = Wrapper(
                MLP(), X_train, y_train, X_test, y_test,
                epochs=epochs, verbose=False, tuning=10,
                damper=damper,
            )
            search = RandomizedSearchCV(
                m, space, n_iter=n_params // 5, n_jobs=80, refit=False, verbose=3,
                random_state=20 + i, cv=[([0], [1, 2]), ([1], [1, 2]), ([2], [0, 1])]
            )
            seeds = 10 + (np.arange(3 * 2) // 2).reshape(3, 2)
            search.fit(seeds.astype(int))

            with open(OUT / "dampersearch" / f"search-{damper}-{i}.pkl", "wb") as f:
                pickle.dump(search, f)
