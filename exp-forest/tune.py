import cloudpickle
import dask.config
import numpy as np
import pandas as pd
import pickle
import random
import warnings
import logging
import torch
import torch.nn.functional as F
from copy import copy
from datetime import datetime
from distributed import Client, as_completed
from distributed.utils_test import log_errors
from functools import partial
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from time import time
from pprint import pprint
from hashlib import sha256
from typing import Any
from dask.cache import Cache

from train import CovClassifier


class Damper(CovClassifier):
    def __init__(
        self, max_batch_size=10_000, weight_decay=1e-4, lr=0.01, device="cpu", seed=42,
        momentum=0.9, opt="sgd",
    ):
        self.max_batch_size = max_batch_size
        super().__init__(lr=lr, seed=seed, device=device, weight_decay=weight_decay, momentum=momentum, opt=opt)

    @property
    def batch_size(self) -> int:
        damping = self.damping()
        self.meta_["damping"] = damping

        # Is the batch size too large? If so, decay the learning rate
        max_bs = self.max_batch_size
        if max_bs is not None and damping >= max_bs:
            self._set_lr(self.lr * max_bs / damping)
            self.meta_["batch_size"] = max_bs
            self.meta_["lr_"] = self.lr * max_bs / damping
            return max_bs

        self.meta_["lr_"] = self.lr
        self.meta_["batch_size"] = damping
        return damping

    def damping(self) -> int:
        return self.max_batch_size

    def _set_lr(self, lr):
        for group in self.model.optimizer_.param_groups:
            group["lr"] = lr
        return self

    def _partial_fit(self, X, y):
        # Get a batch
        bs = self.batch_size
        idx = self.rng_.choice(len(X), size=bs).astype(int)
        data = torch.from_numpy(X[idx].astype("float32")).to(self.device)
        target = torch.from_numpy(y[idx].astype("int64")).to(self.device)

        # Take a step
        self.model.optimizer_.zero_grad()
        output = self.model.module_(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.model.optimizer_.step()

        # Update meta information
        self.meta_["num_examples"] += bs
        self.meta_["model_updates"] += 1
        return self

    def partial_fit(self, X, y):
        if not hasattr(self, "initialized_") or not self.initialized_:
            print("tune.py:85. Re'initing.")
            self.initialize()
        start_eg = self.meta_["num_examples"]
        train_for = len(y)
        while True:
            self._partial_fit(X, y)
            self.history_.append(copy(self.meta_))
            if self.meta_["num_examples"] >= train_for + start_eg:
                break
        return self

    def fit(self, X, y):
        return self

    def score(self, X, y, return_dict=False, prefix=""):
        n_eg = len(y)
        test_loss = 0
        correct = 0
        X_ = np.array_split(X, min(1000, n_eg))
        y_ = np.array_split(y, min(1000, n_eg))
        with torch.no_grad():
            for _x, _y in zip(X_, y_):
                data = torch.from_numpy(_x.astype("float32")).to(self.device)
                target = torch.from_numpy(_y.astype("int64")).to(self.device)
                output = self.model.module_(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= n_eg
        test_acc = correct / n_eg
        if return_dict:
            return {f"{prefix}score": test_acc, f"{prefix}loss": test_loss}
        return test_acc


class PadaDamp(Damper):
    def __init__(
        self,
        initial_batch_size=64,
        batch_growth_rate=0.01,
        dwell=10,
        max_batch_size=10_000,
        seed=None,
        device="cpu",
        lr=0.01,
        weight_decay=1e-6,
        momentum=0.9,
    ):
        self.initial_batch_size = initial_batch_size
        self.batch_growth_rate = batch_growth_rate
        self.dwell = dwell

        super().__init__(
            seed=seed,
            device=device,
            lr=lr,
            max_batch_size=max_batch_size,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    def damping(self) -> int:
        mu = self.meta_["model_updates"]
        if mu % self.dwell == 0:
            b0 = self.initial_batch_size
            bs = b0 + _ceil(self.batch_growth_rate * mu)
            bs_hat = (1 - np.exp(-3e-3 * mu)) * bs
            bs = max(b0 // 4, np.round(bs_hat))
            self.meta_["pada_damping"] = int(bs)

        return self.meta_["pada_damping"]


def _ceil(x: float) -> int:
    return int(np.ceil(x).astype(int))


class GeoDamp(Damper):
    def __init__(
        self,
        initial_batch_size=128,
        dampingdelay=5,
        dampingfactor=2,
        max_batch_size=10_000,
        weight_decay=1e-4,
        seed=42,
        device="cpu",
        lr=0.01,
        momentum=0.9,
    ):
        self.initial_batch_size = initial_batch_size
        self.dampingdelay = dampingdelay
        self.dampingfactor = dampingfactor
        super().__init__(
            seed=seed,
            device=device,
            lr=lr,
            max_batch_size=max_batch_size,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    def damping(self) -> int:
        assert self.dampingfactor >= 1
        epochs = self.meta_["num_examples"] / self.meta_["len_dataset"]
        power = epochs // self.dampingdelay
        factor = max(1, np.power(self.dampingfactor, power))
        damping = int(self.initial_batch_size * factor)
        return damping


class GeoDampLR(GeoDamp):
    def __init__(
        self,
        static_batch_size=128,
        dampingdelay=5,
        dampingfactor=2,
        weight_decay=1e-4,
        seed=42,
        device="cpu",
        lr=0.01,
        momentum=0.9,
    ):
        self.static_batch_size = static_batch_size
        self.dampingdelay = dampingdelay
        self.dampingfactor = dampingfactor
        super().__init__(
            initial_batch_size=static_batch_size,
            max_batch_size=static_batch_size,
            dampingdelay=dampingdelay,
            dampingfactor=dampingfactor,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            lr=lr,
            momentum=momentum,
        )


def _hash(item: Any) -> str:
    b = pickle.dumps(item)
    sha = sha256()
    sha.update(b)
    return sha.hexdigest()


def _get_padadamps(n_params):
    powers = [5, 5.5, 6, 6.5, 7]
    param_space = {
        "initial_batch_size": [2 ** i for i in powers],
        "max_batch_size": [100, 200, 500, 1000, 2000, 5000],
        "batch_growth_rate": loguniform(1e-3, 1e-1),
        "dwell": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        "weight_decay": [1e-3, 1e-4, 1e-5, 1e-6, 0, 0, 0],
    }
    model = PadaDamp(seed=42)

    params = ParameterSampler(param_space, n_iter=n_params, seed=42)
    models = [clone(model).set_params(**p) for p in params]
    return models


def _get_geodamps(n_params):
    model = GeoDamp(seed=42)

    # Don't decay the learning rate:
    # damping delay = 250,000 examples (5 epochs)
    # Tune close to that.
    powers = [5, 6, 7]
    param_space = {
        "initial_batch_size": [2 ** i for i in powers],
        "max_batch_size": [100, 200, 500, 1000, 2000, 5000],
        "dampingfactor": loguniform(1, 10),
        "dampingdelay": loguniform(50e3, 500e3),
        "weight_decay": [1e-3, 1e-4, 1e-5, 1e-6, 0, 0, 0],
    }
    params = ParameterSampler(param_space, n_iter=n_params, seed=42)
    models = [clone(model).set_params(**p) for p in params]
    return models


if __name__ == "__main__":
    N_PARAMS = 200
    N_EPISODES = 1000

    models = _get_padadamps(N_PARAMS) + _get_geodamps(N_PARAMS)
    random.shuffle(models)
    today = datetime.now().isoformat()[:10]

    client = Client("localhost:8786")
    cache = Cache(2e9)  # Leverage two gigabytes of memory
    cache.register()
    print("Submitting jobs...")
    futures = client.map(train, models, n_episodes=N_EPISODES, pure=False)
    print("Done submitting. Writing jobs as they complete:")
    for k, future in enumerate(as_completed(futures)):
        try:
            out = future.result()
            print(k)
            pprint(out)
        except Exception as e:
            print(e)
            continue
        #  ident = out["ident"]
        #  damper_name = out["damper_name"]
        #  score = out["score"]

        #  fname = f"{today}-{damper_name}-{ident}"
        #  print(f"{k}: name={damper_name}, score={score}, ident={ident[:6]}")
        #  with open(f"out/pkl-{fname}.pkl", "wb") as f:
        #  cloudpickle.dump(out, f)

        #  with open(f"out/models-{fname}.pkl", "wb") as f:
        #  cloudpickle.dump(out["models"], f)

        #  pd.DataFrame(out["history"]).to_csv(f"out/csv-{fname}.csv", index=False)
        #  pd.Series(out["params"]).to_json(f"out/params-{fname}.csv")
