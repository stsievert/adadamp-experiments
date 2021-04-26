#  
#  Simulation models for DaskClassifier
#
import dask
import numpy as np
import torch
from torch.optim import Optimizer
from time import sleep, time
from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NewType
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader
from adadamp.adadamp import DaskClassifier

IntArray = Union[List[int], np.ndarray, torch.Tensor]
Number = Union[int, float, np.integer, np.float]
Model = NewType("Model", torch.nn.Module)
Grads = NewType("Grads", Dict[str, Union[torch.Tensor, float, int]])


class DaskClassifierSimulator(DaskClassifier):
    """
    Replaces gradient calculation and scoring functions sleep() functions
    to simulate training scenarios
    """
    
    
    def set_times(self, multiple_workers, score_time, deepcopy_time, grad_time_128): 
        """
        Sets the timings for various aspects of the model
        
        Parameters
        ----------
        multiple_workers : boolean
            Determines is simulator will add additional overhead when simulating
            multiple workers
        score_time : float
            Length of time in seconds the score function should take
        deepcopy_time : float
            Length of time in seconds to spend deepcopying the model to workers
        grad_time_128 : float
            Length of time in seconds to spend per 128 gradiant calculations
        """
        self.mult_machines_ = multiple_workers
        self.score_time = score_time
        self.deepcopy_time = deepcopy_time
        self.grad_time_128 = grad_time_128
    
    def set_sim(self, dic):
        """
        Set the data the simulator should use for batchsize and other parameters
        
        Parameters
        ----------
        dic : dictionary
            Statistics dictionary matching DaskClassifier .meta_ dictionaries
        
        """
        self._sim_data = dic
        self.set_damping(dic["partial_fit__batch_size"])
        
    def set_damping(self, bs: int):
        """
        Set batch size for the simulator
        """
        self.damping_ = bs

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
        Simulates the calculation of gradients for DaskClassifier. Sleeps rather than
        doing actual calculations.

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
        idx = self.random_state_.choice(
            len_dataset, size=min(batch_size, len_dataset), replace=False
        )
        idx.sort()
        worker_idxs = np.array_split(idx, n_workers)

        # Distribute the computation of the gradient. In practice this will
        # mean (say) 4 GPUs to accelerate the gradient computation. Right now
        # for ease it's a small network that doesn't need much acceleration.
        deep_time = self.deepcopy_time
        if self.mult_machines_ == True:
            # DEEPCOPY_TIME = 1.55e-3 * np.log2(P)
            deep_time = self.deepcopy_time * np.log2(self._sim_data['n_workers'])
            
        grads = [
            client.submit(sim_gradient, deep_time, self.grad_time_128, model_opt, dataset, device=device, idx=idx)
            for idx in worker_idxs
        ]

        return grads

    def score(self, X, y=None):
        
        # sleep
        sleep(self.score_time)

        # update internal stats
        stat = {
            "score__acc": self._sim_data["score__acc"],
            "score__loss": self._sim_data["score__loss"],
            "score__time": self.score_time,
        }
        self._meta.update(stat)
        self._meta["score__calls"] += 1

        return stat["score__acc"]


def _randomize(x: torch.Tensor):
    p = np.random.uniform(low=1, high=2)
    return x ** p


def sim_gradient(
    deep_time,
    grad_time,
    model_opt: Tuple[Model, Optimizer],
    train_set,
    *,
    device=torch.device("cpu"),
    idx: IntArray,
    max_bs: int = 1024,
) -> Grads:
    """
    Simulates calculating gradients by sleeping for the time
    it would take to deepcopy the model to a worker and for 
    the time it would theoretically take to compute the gradients
    """

    sleep(deep_time)
    sleep(grad_time * len(idx) / 128)

    model = model_opt[0]

    grads = {k: _randomize(v.data) for k, v in model.named_parameters()}

    return {
        "_num_data": 1,
        "_time": 1,
        "_loss": 1,
        **grads,
    }
