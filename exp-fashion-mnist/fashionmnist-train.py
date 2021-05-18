import os


def _env_prep():
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]:
        os.environ[key] = "1"


_env_prep()

import sys
from pathlib import Path
from pprint import pprint
import itertools
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from time import time
import yaml
import traceback
import sys
import json
from copy import copy

import numpy as np
from distributed import Client, as_completed, LocalCluster
from distributed.scheduler import KilledWorker
from sklearn.model_selection import ParameterSampler

DIR = Path(__file__).absolute().parent
sys.path.append(str(DIR.parent))

import train


def _write(data: List[Dict[str, Any]], filename: str) -> bool:
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return True


def _dummy(**kwargs):
    return [{"time": time(), **kwargs}], [{"train": True}]


def _get_unique(series: pd.Series) -> Any:
    assert series.nunique() == 1
    return series.unique()[0]


if __name__ == "__main__":

    DATA_DIR = (
        #  Path("/mnt")
        #  / "ws"
        #  / "home"
        #  / "sshah"
        #  / "adadamp-experiments"
        #  / "exp-fashion-mnist"
        Path(__file__).parent
        / "_data"
        / datetime.now().isoformat()[:10]
    )
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()
    else:
        d = datetime.now().isoformat()[:10] + "-v2"
        DATA_DIR = DATA_DIR.parent / d
        DATA_DIR.mkdir()
    print(f"DATA_DIR = {DATA_DIR}")
    x = input("Continue? y/n : ")
    assert x.lower() in ["y", "n"]
    if x.lower() == "n":
        sys.exit(1)

    epochs = 200
    n_runs = 1
    seed_start = 1000

    #  with open("hyperparams.json", "r") as f:
    #  params = json.load(f)
    #  for damp, param in params.items():
    #  param["damper"] = damp

    print("n_runs =", n_runs)
    cont = input("Ok? y/n : ")
    if cont.lower() == "n":
        sys.exit(1)

    cluster = LocalCluster(n_workers=60, threads_per_worker=1, processes=True)
    client = Client(cluster)

    def _prep():
        _env_prep()
        import torch

        torch.set_num_threads(1)
        return True

    client.run(_prep)

    def submit(seed, **kwargs):
        import train

        # assert train.__version__ == "0.1"

        import adadamp

        assert adadamp.__version__ == "0.2.0rc4"

        return train.main(
            epochs=epochs,
            verbose=False,
            init_seed=seed,
            random_state=seed,
            tuning=True,
            **kwargs,
        )

    futures = []
    seeds = np.arange(seed_start, seed_start + n_runs)
    dampers = ["geodamp", "adagrad", "geodamplr", "radadamp"]
    #  assert set(dampers).issubset(set(params.keys()))

    param_dist = {
        "lr": [0.005],
        "momentum": [0.9],
        "nesterov": [True],
        "rho": [0.95, 0.99, 0.995, 0.999],
        "dampingfactor": np.logspace(0, 1, num=1000),
        "dampingdelay": [2, 5, 10, 20, 30, 60],
        "initial_batch_size": [16, 32, 64, 128, 256],
        "max_batch_size": [256, 512, 1024, 4096],
        "weight_decay": [3e-3, 1e-3, 3e-4, 1e-4],
        "dwell": [1, 10, 100, 300, 1000, 3000],
    }
    n_iters = {
        "radadamp": 30,
        "geodamp": 30,
        "adagrad": 6,
    }
    for damper in dampers:
        if damper not in n_iters:
            continue
        damper_params = ParameterSampler(param_dist, n_iter=n_iters[damper])
        for params in damper_params:
            futures.extend(client.map(submit, seeds, damper=damper, **params))

    for k, future in enumerate(as_completed(futures)):
        try:
            data, train_data = future.result()
        except:  # KilledWorker:
            # This is likely a problem with my code rather than with the
            # Dask cluster.
            #
            # https://stackoverflow.com/questions/46691675/what-do-killedworker-exceptions-mean-in-dask
            print("-" * 20)
            for info in sys.exc_info():
                print(info)
        else:
            print(f"writing training {k}")
            _write(data, str(DATA_DIR / f"{k}-test.csv.zip"))
            _write(train_data, str(DATA_DIR / f"{k}-train.csv.zip"))
