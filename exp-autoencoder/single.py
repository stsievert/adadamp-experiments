
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
import os

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

CUDA_VISIBLE_DEVICES = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
print(f"{CUDA_VISIBLE_DEVICES=}")

DIR = Path(__file__).absolute().parent
DS_DIR = DIR / "dataset"
OUT = DIR / "data-tuning"
sys.path.extend([str(DIR.parent), str(DIR.parent / "train")])

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

    def fit(self, X=None, y=None):
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
        fname = f"{ident}-{seed}-{CUDA_VISIBLE_DEVICES}.pkl.zip"

        kwargs = params
        if "damplr" in kwargs["damper"] or "dampnnlr" in kwargs["damper"]:
            kwargs["damper"] = kwargs["damper"][:-2]
            kwargs["max_batch_size"] = self.initial_batch_size

        out_dir = OUT / "dampersweep"
        if out_dir / fname in out_dir.iterdir():
            print(f"Skipping {fname}, already in directory", flush=True)
            self.df_ = pd.read_pickle(out_dir / fname)
            return self

        start = time()
        try:
            data, train_data, model, test_set = train.main(
                dataset="autoencoder",
                model=None,
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

if __name__ == "__main__":
    base_search_space = {
        "lr": loguniform(1e-4, 1e-2),
        "initial_batch_size": [2**i for i in range(4, 9 + 1)],
        "max_batch_size": [2**i for i in range(7, 13 + 1)],
        "dwell": loguniform(1, 1e3),
        "wait": loguniform(1, 1e3),
        "weight_decay": loguniform(1e-7, 1e-4),
        "momentum": loguniform_m1(1e-1, 1e-3),
        "nesterov": [True],
    }
    static_ibs = [2**i for i in range(5, 9 + 1)]
    def rv_bounds(*, low, high):
        assert low < high, f"{low=} > {high=}"
        return low, high - low
    damper_search_space: Dict[str, Dict[str, Any]] = {
        "adagrad": {
            "lr": loguniform(1e-4, 1e-2),
            "initial_batch_size": static_ibs,
        },
        "adamw": {
            "lr": loguniform(1e-3, 1e-1),
            "weight_decay": loguniform(1e-3, 1e-1),
            "initial_batch_size": static_ibs,
        },
        "nadam": {
            "lr": loguniform(0.5e-3, 1e-2),
            "initial_batch_size": static_ibs,
        },
        "geodamp": {
            "dampingdelay": ints(1, 10),
            "dampingfactor": ints(1, 21),
        },
        "radadamp": {
            "reduction": ["min", "mean"], # "median"],
            "rho": uniform(rv_bounds(low=0.2, high=0.99)),
            "momentum": uniform(rv_bounds(low=0.5, high=0.999)),
        },
        "padadamp": {
            "lr": loguniform(0.5e-4, 1e-2),
            "growth_rate": loguniform(1e-3, 1e2),
            "momentum": uniform(rv_bounds(low=0.5, high=0.999)),
        },
    }
    for damper in ["radadamp", "padadamp", "geodamp"]:
        damper_search_space[f"{damper}lr"] = damper_search_space[damper]

    # LR      |BS      |damper
    # passive |static  |geodamplr
    # static  |passive |geodamp
    # static  |static  |GD, SGD
    # passive |passive |geodamp (small MBS)
    # adaptive|static  |adamw
    # static  |adaptive|radadamp
    # adaptive|adaptive|adadagrad, radadamp (small MBS)
    #
    # adaptive and passive aren't mixed

    for i in range(100):
        for damper in [
            "geodamp",
            "padadamp",
            "radadamp",
            "geodamplr",
            "adamw",
            "adagrad",
            "padadamplr",
            "radadamplr",
            "gd",
            "sgd",
        ]:
            print(f"\n\n## {i}th tuning run for {damper}\n\n")
            space = deepcopy(base_search_space)
            space.update(damper_search_space.get(damper, {}))
            space.update({"damper": [damper]})

            epochs = 100 if damper != "gd" else 1000

            seeds = 12 + (np.arange(3 * 2) // 2).reshape(3, 2)
            m = Wrapper(
                epochs=epochs, verbose=False, tuning=11, damper=damper,
            )
            n_jobs = 10
            search = RandomizedSearchCV(
                m, space, n_iter=n_jobs, n_jobs=n_jobs,
                refit=False, verbose=3,
                random_state=24 + i + CUDA_VISIBLE_DEVICES, cv=[([0], [1, 2])],
            )
            search.fit(seeds.astype(int))

            with open(OUT / "dampersearch" / f"search-{damper}-{i}-{CUDA_VISIBLE_DEVICES}.pkl", "wb") as f:
                pickle.dump(search, f)
