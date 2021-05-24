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
from copy import copy, deepcopy

import numpy as np
from distributed import Client, as_completed, LocalCluster
from distributed.scheduler import KilledWorker
from sklearn.model_selection import ParameterSampler


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
    n_runs = 20
    seed_start = 1000

    print("n_runs =", n_runs)
    cont = input("Ok? y/n : ")
    if cont.lower() == "n":
        sys.exit(1)

    client = Client("localhost:8786")

    client.upload_file("wideresnet.py")
    client.upload_file("train.py")

    def _prep():
        _env_prep()
        import torch

        torch.set_num_threads(1)
        return True

    client.run(_prep)

    def submit(seed, **kwargs):
        import adadamp
        assert adadamp.__version__ == "0.2.0rc6"

        return train.main(
            epochs=epochs,
            verbose=True,
            init_seed=seed,
            random_state=seed,
            tuning=True,
            **kwargs,
        )

    futures = []
    seeds = np.arange(seed_start, seed_start + n_runs)

    with open("hyperparams.json", "r") as f:
        paramslt = json.load(f)
    for i in paramslt:
        paramslt[i]["damper"] = i
    paramslt["geodamplr"] = deepcopy(paramslt["geodamp"])
    paramslt["geodamplr"]["max_batch_size"] = paramslt["geodamplr"]["initial_batch_size"]
    paramslt["geodamplr"]["damper"] = "geodamplr"
    paramslt["radadamplr"] = deepcopy(paramslt["radadamp"])
    paramslt["radadamplr"]["max_batch_size"] = paramslt["radadamplr"]["initial_batch_size"]
    paramslt["radadamplr"]["damper"] = "radadamp"

    for params in paramslt.values():
        futures.extend(client.map(submit, seeds, **params))

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
            print(f"writing training {k} to {DATA_DIR}")
            _write(data, str(DATA_DIR / f"{k}-test.csv.zip"))
            _write(train_data, str(DATA_DIR / f"{k}-train.csv.zip"))
