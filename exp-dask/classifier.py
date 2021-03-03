from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NewType
from warnings import warn

import dask
import dask.array as da
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.optim import Optimizer
from time import sleep, time
from distributed import get_client
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from torch.autograd import Variable
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader

from adadamp.adadamp import DaskClassifier

IntArray = Union[List[int], np.ndarray, torch.Tensor]
Number = Union[int, float, np.integer, np.float]
Model = NewType("Model", torch.nn.Module)
Grads = NewType("Grads", Dict[str, Union[torch.Tensor, float, int]])



SCORE_TIME = 0.0  # 5.39407711148262
DEEPCOPY_TIME = 0.0  # 0.05855  # seconds
GRAD_TIME_128 = 0.0  # 0.07832  # seconds


class DaskClassifierExpiriments(DaskClassifier):
    def set_damping(self, bs: int):
        self._batch_size = bs

    def batch_size_(self) -> int:
        return self._batch_size

class DaskClassifierSimulator(DaskClassifierExpiriments):
    
    def set_times(self, score_time, deepcopy_time, grad_time_128):
        global SCORE_TIME
        global DEEPCOPY_TIME
        global GRAD_TIME_128
        
        SCORE_TIME = score_time
        DEEPCOPY_TIME = deepcopy_time
        GRAD_TIME_128 = grad_time_128
    
    def set_sim(self, dic):
        """
        Sets simulation data for next epoch
        """
        self._sim_data = dic
        self.set_damping(dic["partial_fit__batch_size"])

    def _get_gradients(
        self,
        start_idx,
        model_opt,
        dataset,
        *,
        loss,
        batch_size,
        client,
        n_workers,
        len_dataset,
        device,
    ):
        """
        Calculates the gradients at a given state. This function is
        mainly in charge of sending the gradient calculation off to the
        dask workers, the hard work is done in _dist.gradient()

        Note:
        - In this implementation, each worker manually grabs the data
          and caches it themselves. This is a work around for a dask bug
        - See the doc string of _get_fashionminst in _dist.py for more info

        Parameters
        ----------
        start_idx : int
            Index to start sampling at. Data indices from ``start_idx`` to
            ``start_idx + batch_size`` will be sampled.

        model_future : distributed.future (?)
            Future object of the model to get gradients for

        n : int
            number of items in training set

        idx: int
            Current epoch, used to set up random state

        batch_size: int
            Current batch size

        n_workers: int
            Number of workers current running

        """
        # Iterate through the dataset in batches
        # TODO: integrate with IterableDataset (this is pretty much already
        # an IterableDataset but without vectorization)
        idx = self.random_state_.choice(
            len_dataset, size=min(batch_size, len_dataset), replace=False
        )
        idx.sort()
        worker_idxs = np.array_split(idx, n_workers)

        # Distribute the computation of the gradient. In practice this will
        # mean (say) 4 GPUs to accelerate the gradient computation. Right now
        # for ease it's a small network that doesn't need much acceleration.

        grads = [
            client.submit(sim_gradient, 0, model_opt, dataset, device=device, idx=idx)
            for idx in worker_idxs
        ]

        return grads

    def score(self, X, y=None):

        # sleep
        score_time = SCORE_TIME
        sleep(score_time)

        # update internal stats
        stat = {
            "score__acc": self._sim_data["score__acc"],
            "score__loss": self._sim_data["score__loss"],
            "score__time": SCORE_TIME,
        }
        self._meta.update(stat)
        self._meta["score__calls"] += 1

        return stat["score__acc"]


def _randomize(x: torch.Tensor):
    p = np.random.uniform(low=1, high=2)
    return x ** p


def sim_gradient(
    timing,
    model_opt: Tuple[Model, Optimizer],
    train_set,
    *,
    device=torch.device("cpu"),
    idx: IntArray,
    max_bs: int = 1024,
) -> Grads:

    sleep(DEEPCOPY_TIME)
    sleep(GRAD_TIME_128 * len(idx) / 128)

    model = model_opt[0]

    grads = {k: _randomize(v.data) for k, v in model.named_parameters()}

    return {
        "_num_data": 1,
        "_time": 1,
        "_loss": 1,
        **grads,
    }
