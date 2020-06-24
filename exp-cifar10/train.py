import sys
from pathlib import Path
import itertools
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import numpy as np
from time import time
import yaml
import traceback
import sys
import os
import random
import json
from copy import copy

from pprint import pprint
from time import time, sleep
from distributed import Client, as_completed

import train
from train.train import _test


def _get_unique(x: pd.Series) -> Any:
    assert x.nunique() == 1
    return x.iloc[0]

def job(kwds):
    import train
    assert train.__version__ == "0.1.5"

    import adadamp
    assert adadamp.__version__ == "0.1.8"
    return train.main(**kwds)


if __name__ == "__main__":
    epochs = 210  # 210 epochs total
    TUNERS = [False]
    INIT_SEEDS = [91]
    RANDOM_STATES = [220]

    DIR = Path(__file__).parent.absolute()
    with open("chosen-params.yaml", "r") as _:
        params = yaml.load(_, Loader=yaml.SafeLoader)
    params = [{"damper": damper, **p} for damper, p in params.items()]

    today = datetime.now().isoformat()[:10]
    DATA_DIR = DIR / "train-data"
    if DATA_DIR.exists():
        files = list(DATA_DIR.iterdir())
        if len(files):
            DATA_DIR = DATA_DIR.parent / f"{today}-v2"
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()

    print(f"DATA_DIR={DATA_DIR}")
    print(f"epochs={epochs}")
    print(f"INIT_SEEDS={INIT_SEEDS}")

    # Tuning sets are the same for every model.
    static = {"epochs": epochs}
    repeat_kwargs = [
        {"tuning": tuning, "init_seed": init_seed, "random_state": rs, **static}
        for init_seed, (tuning, rs) in itertools.product(
            INIT_SEEDS, zip(TUNERS, RANDOM_STATES)
        )
    ]
    kwargs = [
        {**repeat_kwarg, **param}
        for repeat_kwarg in repeat_kwargs
        for param in run
    ]

    # Pass (tuning, param_seed, init_seed) to train.choose_params_and_run
    files = [f.name for f in DATA_DIR.glob("*.parquet")]
    print("Jobs to submit (before filtering):", len(kwargs))
    print("Completed jobs:", len(files) / 2)
    print("Submitting {} jobs".format(len(kwargs)))
    print("kwargs =")
    pprint(kwargs)
    ans = input("Continue? (y/N) : ")
    if ans.lower() != "y":
        sys.exit(1)

    ## Launch with this command for g4dn.xlarge:
    ## dask-worker --nprocs 2 --nthreads 1 --resources "GPUMEM=1" localhost:8786
    client = Client("localhost:8786")
    print(f"Submitting {len(kwargs)} jobs to {TAG}...")
    futures = client.map(job, kwargs, resources={"GPUMEM": 1})

    print(f"Done submitting jobs")

    for k, future in enumerate(as_completed(futures)):
        data, train_data = future.result()

        df = pd.DataFrame(data)
        train_df = pd.DataFrame(train_data)
        ident = _get_unique(df["ident"])
        tuning = _get_unique(df["tuning"])
        damper = _get_unique(df["damper"])
        init_seed = _get_unique(df["init_seed"])
        rs = _get_unique(df["random_state"])
        fname = f"{damper}-{init_seed}-{rs}-{tuning}-{ident}"
        print(k, fname)
        df.to_parquet(str(DATA_DIR / f"{fname}-test.parquet"), index=False)
        train_df.to_parquet(str(DATA_DIR / f"{fname}-train.parquet"), index=False)