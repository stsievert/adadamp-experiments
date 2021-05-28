from __future__ import print_function
import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[key] = "1"
from types import SimpleNamespace
from typing import Any, Dict, List, Union, Optional, Tuple
import hashlib
import pickle
from pprint import pprint
import itertools
from copy import copy
from datetime import datetime
import random
from time import time
import os
import sys

import numpy.linalg as LA

from packaging import version
import numpy as np
import yaml
import pandas as pd
import torch
import scipy.stats
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import ParameterSampler
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
import torchvision.models as models
import torch.utils.data

torch.set_num_threads(1)

from adadamp import (
    AdaDamp,
    GeoDamp,
    PadaDamp,
    RadaDamp,
    BaseDamper,
    GeoDampLR,
    CntsDampLR,
    GradientDescent,
)
from adadamp.utils import _get_resnet18
import adadamp.experiment as experiment
from .wideresnet import WideResNet

Number = Union[float, int, np.float, np.int]
TensorDataset = torch.utils.data.TensorDataset
if version.parse(scipy.__version__) <= version.parse("1.4.0"):
    # See https://github.com/scipy/scipy/pull/10815
    loguniform = scipy.stats.reciprocal
else:
    loguniform = scipy.stats.loguniform


class Net(nn.Module):
    """
    111k params, ~150s/epoch
    """

    def __init__(self):
        super(Net, self).__init__()
        self.hidden_size = 100
        self.final_convs = 100
        self.conv1 = nn.Conv2d(1, 30, 5, stride=1)
        self.conv2 = nn.Conv2d(30, 60, 5, stride=1)
        self.conv3 = nn.Conv2d(60, self.final_convs, 3, stride=1)
        self.fc1 = nn.Linear(1 * 1 * self.final_convs, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 1 * 1 * self.final_convs)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LinearNet(nn.Module):
    def __init__(self, d):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.fc3 = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.reshape(-1)


def stable_hash(w: bytes) -> str:
    h = hashlib.sha256(w)
    return h.hexdigest()


def _get_stable_val(x: Any) -> Any:
    if isinstance(x, (float, np.float)):
        y = np.round(x, decimals=5)
        return int(y * 10 ** 5)
    return x


def ident(args: dict) -> str:
    ordered_keys = sorted(list(args.keys()))
    v = [
        (
            k,
            _get_stable_val(args[k])
            if not isinstance(args[k], dict)
            else ident(args[k]),
        )
        for k in ordered_keys
    ]
    return str(stable_hash(pickle.dumps(v)))


def _set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return True


def main(
    dataset: str = "fashionmnist",
    initial_batch_size: int = 64,
    epochs: int = 6,
    verbose: Union[int, bool] = False,
    lr: float = 1.0,
    cuda: bool = False,
    random_state: Optional[int] = None,  # seed to pass to BaseDamper
    init_seed: Optional[int] = None,  # seed for initialization
    tuning: bool = True,  # tuning seed
    damper: str = "geodamp",
    batch_growth_rate: float = 0.01,
    dampingfactor: Number = 5.0,
    dampingdelay: int = 5,
    max_batch_size: Optional[int] = None,
    test_freq: float = 1,
    approx_loss: bool = False,
    rho: float = 0.9,
    dwell: int = 1,
    approx_rate: bool = False,
    model: Optional[str] = None,
    momentum: Optional[Union[float, int]] = 0,
    nesterov: bool = False,
    weight_decay: float=0,
) -> Tuple[List[Dict], List[Dict]]:
    # Get (tuning, random_state, init_seed)
    assert isinstance(tuning, (bool, int))
    assert isinstance(random_state, (int,np.integer,np.int64))
    assert isinstance(init_seed, (int,np.integer,np.int64))

    torch.set_num_threads(1)

    args: Dict[str, Any] = {
        "initial_batch_size": initial_batch_size,
        "max_batch_size": max_batch_size,
        "batch_growth_rate": batch_growth_rate,
        "dampingfactor": dampingfactor,
        "dampingdelay": dampingdelay,
        "epochs": epochs,
        "verbose": verbose,
        "lr": lr,
        "no_cuda": not cuda,
        "random_state": random_state,
        "init_seed": init_seed,
        "damper": damper,
        "dataset": dataset,
        "approx_loss": approx_loss,
        "test_freq": test_freq,
        "rho": rho,
        "dwell": dwell,
        "approx_rate": approx_rate,
        "nesterov": nesterov,
        "momentum": momentum,
        "weight_decay": weight_decay,
    }
    pprint(args)

    no_cuda = not cuda
    args["ident"] = ident(args)
    args["tuning"] = tuning

    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    _device = torch.device(device)
    _set_seed(args["init_seed"])

    transform_train = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ]
    transform_test = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    assert dataset in ["fashionmnist", "cifar10", "synthetic", "mnist"]
    if dataset == "fashionmnist":
        _dir = "_traindata/fashionmnist/"
        train_set = FashionMNIST(
            _dir, train=True, transform=Compose(transform_train), download=True,
        )
        test_set = FashionMNIST(_dir, train=False, transform=Compose(transform_test))
        model = Net()
    elif dataset == "mnist":
        _dir = "_traindata/mnist/"
        train_set = MNIST(
            _dir, train=True, transform=Compose(transform_train), download=True,
        )
        test_set = MNIST(_dir, train=False, transform=Compose(transform_test))
        model = Net()
    elif dataset == "cifar10":
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

        _dir = "_traindata/cifar10/"
        train_set = CIFAR10(
            _dir, train=True, transform=Compose(transform_train), download=True,
        )
        test_set = CIFAR10(_dir, train=False, transform=Compose(transform_test))
        if model == "wideresnet":
            model = WideResNet(16, 4, 0.3, 10)
        else:
            model = _get_resnet18()
    elif dataset == "synthetic":
        data_kwargs = {"n": 10_000, "d": 100}
        args.update(data_kwargs)
        train_set, test_set, data_stats = synth_dataset(**data_kwargs)
        args.update(data_stats)
        model = LinearNet(data_kwargs["d"])
    else:
        raise ValueError(
            f"dataset={dataset} not in ['fashionmnist', 'cifar10', 'synth', 'mnist']"
        )
    if tuning:
        train_size = int(0.8 * len(train_set))
        test_size = len(train_set) - train_size

        train_set, test_set = random_split(
            train_set, [train_size, test_size], random_state=int(tuning),
        )
        train_x = [x.abs().sum().item() for x, _ in train_set]
        train_y = [y for _, y in train_set]
        test_x = [x.abs().sum().item() for x, _ in test_set]
        test_y = [y for _, y in test_set]
        data_stats = {
            "train_x_sum": sum(train_x),
            "train_y_sum": sum(train_y),
            "test_x_sum": sum(test_x),
            "test_y_sum": sum(test_y),
            "len_train_x": len(train_x),
            "len_train_y": len(train_y),
            "len_test_x": len(test_x),
            "len_test_y": len(test_y),
            "tuning": int(tuning),
        }
        args.update(data_stats)
        pprint(data_stats)

    model = model.to(_device)
    _set_seed(args["random_state"])

    if args["damper"] == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.get("lr", 0.01))
    elif args["damper"] == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), rho=rho)
    else:
        if not args["nesterov"]:
            print("Use nesterov momentum")
            assert args["momentum"] == 0
        optimizer = optim.SGD(model.parameters(), lr=args["lr"], nesterov=args["nesterov"], momentum=args["momentum"], weight_decay=args["weight_decay"])
    n_data = len(train_set)

    opt_args = [model, train_set, optimizer]
    opt_kwargs = {k: args[k] for k in ["initial_batch_size", "max_batch_size", "random_state"]}
    opt_kwargs["device"] = device
    if dataset == "synthetic":
        opt_kwargs["loss"] = F.mse_loss
    if dataset == "cifar10":
        opt_kwargs["loss"] = F.cross_entropy
    if args["damper"].lower() == "padadamp":
        if approx_rate:
            assert isinstance(max_batch_size, int)
            BM = max_batch_size
            B0 = initial_batch_size
            e = epochs
            n = n_data
            r_hat = 4/3 * (BM - B0) * (B0 + 2*BM + 3)
            r_hat /= (2*BM - 2*B0 + 3 * e * n)
            args["batch_growth_rate"] = r_hat

        opt = PadaDamp(
            *opt_args,
            batch_growth_rate=args["batch_growth_rate"],
            dwell=args["dwell"],
            **opt_kwargs,
        )
    elif args["damper"].lower() == "geodamp":
        opt = GeoDamp(
            *opt_args,
            dampingdelay=args["dampingdelay"],
            dampingfactor=args["dampingfactor"],
            **opt_kwargs,
        )
    elif args["damper"].lower() == "radadamp":
        opt = RadaDamp(*opt_args, **opt_kwargs)
    elif args["damper"].lower() == "geodamplr":
        opt = GeoDampLR(
            *opt_args,
            dampingdelay=args["dampingdelay"],
            dampingfactor=args["dampingfactor"],
            **opt_kwargs,
        )
    elif args["damper"].lower() == "cntsdamplr":
        opt = CntsDampLR(*opt_args, dampingfactor=args["dampingfactor"], **opt_kwargs,)
    elif args["damper"].lower() == "adadamp":
        opt = AdaDamp(
            *opt_args, approx_loss=approx_loss, dwell=args["dwell"], **opt_kwargs
        )
    elif args["damper"].lower() == "gd":
        opt = GradientDescent(*opt_args, **opt_kwargs)
    elif (
        args["damper"].lower() in ["adagrad", "adadelta", "sgd", "gd"]
        or args["damper"] is None
    ):
        opt = BaseDamper(*opt_args, **opt_kwargs)
    else:
        raise ValueError("argument damper not recognized")
    if dataset == "synthetic":
        pprint(data_stats)
        opt._meta["best_train_loss"] = data_stats["best_train_loss"]

    data, train_data = experiment.run(
        model=model,
        opt=opt,
        train_set=train_set,
        test_set=test_set,
        args=args,
        test_freq=test_freq,
        train_stats=dataset == "synthetic",
        verbose=verbose,
        device="cuda" if use_cuda else "cpu",
    )
    return data, train_data


def synth_dataset(
    n: int = 1000, d: int = 100
) -> Tuple[TensorDataset, TensorDataset, Dict[str, float]]:
    rng = np.random.RandomState(42)
    A = rng.normal(size=(n, d)).astype("float32")
    x_star = rng.normal(size=d).astype("float32")
    sigma = d * 0.1  # AdaDamp's loss goes negative when 0.2 * d used?
    sigma = (d / 10) * 0.1
    y = A @ x_star + sigma * rng.normal(size=n).astype("float32")

    p_test = 0.2
    train_idx = rng.choice([False, True], size=len(y), p=[p_test, 1 - p_test]).astype(
        bool
    )
    A_train, y_train = A[train_idx], y[train_idx]
    A_test, y_test = A[~train_idx], y[~train_idx]

    x_hat_star = LA.inv(A_train.T @ A_train) @ A_train.T @ y_train
    train_losses = (A_train @ x_hat_star - y_train) ** 2
    test_losses = (A_test @ x_hat_star - y_test) ** 2

    A_train = torch.from_numpy(A_train)
    y_train = torch.from_numpy(y_train)
    A_test = torch.from_numpy(A_test)
    y_test = torch.from_numpy(y_test)
    train_set = torch.utils.data.TensorDataset(A_train, y_train)
    test_set = torch.utils.data.TensorDataset(A_test, y_test)
    return (
        train_set,
        test_set,
        {"best_train_loss": train_losses.mean(), "best_test_loss": test_losses.mean()},
    )


def choose_params_and_run(parameters=None, param_seed=0, damper="sgd", **kwargs):
    # Get (tuning, param_seed, init_seed, damper)
    params = parameters.pop("base")
    params.update(parameters[damper])
    for k, p in params.items():
        if isinstance(p, dict):
            if p["space"] == "log":
                low, high = p["base"] ** p["low"], p["base"] ** p["high"]
                params[k] = loguniform(low, high)
            elif p["space"] == "linear":
                params[k] = scipy.stats.uniform(p["low"], p["high"] - p["low"])
            if "type" in p:
                params[k] = params[k][p["type"]]
    param = list(ParameterSampler(params, n_iter=1, random_state=param_seed))[0]
    msg = "Don't double specify the parameters"
    assert set(params.keys()).intersection(set(kwargs.keys())) == set(), msg
    assert all(
        c not in params.keys() and c not in kwargs.keys() for c in ["damper", "param_seed"]
    )
    # Pass (tuning, random_state, init_seed) to main
    return main(damper=damper, random_state=param_seed, **param, **kwargs)



def random_split(dataset, lengths, random_state=None):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced

    (note: slightly modified from torch.utils.data.random_split)
    """
    from torch._utils import _accumulate
    from torch.utils.data import Subset

    rng = np.random.RandomState(seed=random_state)
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = rng.permutation(sum(lengths)).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def _test(**kwds):
    return {
        "_hash": hash("adsjfk;l42"),
        "version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        **kwds,
    }


if __name__ == "__main__":
    r = main(
        dataset="fashionmnist",
        initial_batch_size=2048,
        epochs=4,
        verbose=5000,
        lr=1e-2,
        damper="adagrad",
        tuning=True,
        test_freq=1,
        cuda=True,
    )
