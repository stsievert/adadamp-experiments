from __future__ import print_function
from types import SimpleNamespace
from typing import Any, Dict, List, Union, Optional, Tuple
import hashlib
import pickle
import itertools
from copy import copy
from datetime import datetime
import random
from time import time
import os

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
from torchvision.transforms import v2
from torchvision.datasets import FashionMNIST, CIFAR10
import torchvision
import torchvision.models as models
import torch.utils.data
from torch.utils.data.dataset import Dataset

from adadamp import (
    AdaDamp,
    GeoDamp,
    PadaDamp,
    BaseDamper,
    GeoDampLR,
    CntsDampLR,
    GradientDescent,
    RadaDamp,
)
from adadamp.utils import _get_resnet18
import adadamp.experiment as experiment

Number = Union[float, int, np.integer]
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

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Decoder.

        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

class Autoencoder(nn.Module):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        num_input_channels: int = 3,
    ):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim)
        self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

def stable_hash(w: bytes) -> str:
    h = hashlib.sha256(w)
    return h.hexdigest()


def _get_stable_val(x: Any) -> Any:
    if isinstance(x, (float, np.float64)):
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return True


class NoisyImages(torchvision.datasets.SVHN):  # good; ~3 bits => can be compressed
#class NoisyImages(torchvision.datasets.CIFAR10):  # no one can criticize this
#class NoisyImages(torchvision.datasets.Caltech256):  # too long
    def __getitem__(self, *args, **kwargs):
        image, target = super().__getitem__(*args, **kwargs)
        return image, image

def scale(x, bounds=(-1, 1)):
    width = bounds[1] - bounds[0]
    offset = bounds[0]
    return width*x + offset

def main(
    dataset: str = "fashionmnist",
    initial_batch_size: int = 64,
    epochs: int = 6,
    verbose: Union[int, bool] = False,
    lr: float = 1.0,
    cuda: bool = True,
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
    train_data=None, test_data=None,
) -> Tuple[List[Dict], List[Dict], nn.Module, Dataset]:
    # Get (tuning, random_state, init_seed)
    assert int(tuning) or isinstance(tuning, bool)
    assert isinstance(random_state, int)
    assert isinstance(init_seed, int)

    if "NUM_THREADS" in os.environ:
        v = os.environ["NUM_THREADS"]
        if v:
            print(f"NUM_THREADS={v} (int(v)={int(v)})")
            torch.set_num_threads(int(v))

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

    no_cuda = not cuda
    args["ident"] = ident(args)
    args["tuning"] = tuning

    use_cuda = cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    _device = torch.device(device)
    _set_seed(args["init_seed"])

    mean = 0.1307
    std = 0.3081
    transform_train = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean,), std=(std,)),
    ]
    transform_test = [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    itransform = transforms.Normalize(mean=(-mean / std), std=1 / std)
    print("302", flush=True)
    if dataset == "fashionmnist":
        _dir = "_traindata/fashionmnist/"
        train_set = FashionMNIST(
            _dir, train=True, transform=Compose(transform_train), download=True,
        )
        test_set = FashionMNIST(_dir, train=False, transform=Compose(transform_test))
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
            from .wideresnet import WideResNet
            model = WideResNet(16, 4, 0.3, 10)
        else:
            model = _get_resnet18()
    elif dataset == "synthetic":
        data_kwargs = {"n": 10_000, "d": 100}
        args.update(data_kwargs)
        train_set, test_set, data_stats = synth_dataset(**data_kwargs)
        args.update(data_stats)
        model = LinearNet(data_kwargs["d"])
    elif dataset == "autoencoder":
        # policy = v2.AutoAugmentPolicy.SVHN

        transform_train = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(),
            v2.RandomGrayscale(),
            v2.GaussianNoise(sigma=0.01),
            v2.Lambda(scale),
        ]
        transform_test = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(scale),
        ]

        _dir = "_traindata/svhn/"
        #train_set = NoisyImages(
        #    _dir, train=True, transform=v2.Compose(transform_train), download=True,
        #)
        #test_set = NoisyImages(_dir, train=False, transform=v2.Compose(transform_test), download=True)
        train_set = NoisyImages(
            _dir, split="train", transform=v2.Compose(transform_train), download=True,
        )
        test_set = NoisyImages(_dir, split="test", transform=v2.Compose(transform_test), download=True)
        model = Autoencoder(32, 100, num_input_channels=3)
        n_params = sum(p.detach().numpy().size for p in model.parameters())
        print(f"n_params = {n_params / 1e6:0.1f}M")
        # 220M parameters with (32 * 32, 200, n_input=3)
        # 65M parameters with 32 * 16, 400, n_input=3
        # 400K params quick and good
    elif train_data is not None and test_data is not None:
        train_set = TensorDataset(*train_data)
        test_set = TensorDataset(*test_data)
    else:
        raise ValueError(
            f"dataset={dataset} not in ['fashionmnist', 'cifar10', 'synth']"
        )
    print("350", flush=True)
    if tuning:
        train_size = int(0.8 * len(train_set))
        test_size = len(train_set) - train_size

        train_set, test_set = random_split(
            train_set, [train_size, test_size], random_state=int(tuning),
        )
        test_set.dataset.transform=v2.Compose(transform_test)
        train_x = [x.abs().sum().item() for x, _ in train_set]
        train_y = [y.abs().sum().item() for _, y in train_set]
        test_x = [x.abs().sum().item() for x, _ in test_set]
        test_y = [y.abs().sum().item() for _, y in test_set]
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
        print(data_stats)

    model = model.to(_device)
    _set_seed(args["random_state"])

    if args["damper"] == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.get("lr", 0.01))
    elif args["damper"] == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), rho=rho)
    else:
        if not args["nesterov"]:
            assert args["momentum"] == 0
        optimizer = optim.SGD(model.parameters(), lr=args["lr"], nesterov=args["nesterov"], momentum=args["momentum"], weight_decay=args["weight_decay"])
    n_data = len(train_set)

    opt_args = [model, train_set, optimizer]
    opt_kwargs = {k: args[k] for k in ["initial_batch_size", "max_batch_size", "random_state"]}
    opt_kwargs["device"] = device
    if dataset in ["synthetic", "autoencoder"]:
        opt_kwargs["loss"] = F.mse_loss
    elif dataset == "cifar10":
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
    if args["damper"].lower() == "adadampnn":
        print("458 adadamnnn", flush=True)
        opt = AdaDampNN(
            *opt_args,
            batch_growth_rate=args["batch_growth_rate"],
            dwell=args["dwell"],
            **opt_kwargs,
        )
    elif args["damper"].lower() == "radadamp":
        opt = RadaDamp(*opt_args, rho=rho, **opt_kwargs)
    elif args["damper"].lower() == "geodamp":
        opt = GeoDamp(
            *opt_args,
            dampingdelay=args["dampingdelay"],
            dampingfactor=args["dampingfactor"],
            **opt_kwargs,
        )
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
        raise ValueError(f"argument damper={damper} not recognized")
    if dataset == "synthetic":
        print(data_stats)
        opt._meta["best_train_loss"] = data_stats["best_train_loss"]

    data, train_data = experiment.run(
        model=model,
        opt=opt,
        train_set=train_set,
        test_set=test_set,
        args=args,
        test_freq=test_freq,
        train_stats=True,#dataset == "synthetic",
        verbose=verbose,
        device="cuda" if use_cuda else "cpu",
    )
    return data, train_data, model, test_set


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



def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

def random_split(dataset, lengths, random_state=None):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced

    (note: slightly modified from torch.utils.data.random_split)
    """
    # from torch._utils import _accumulate
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
