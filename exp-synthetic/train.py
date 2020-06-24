import sys
from pathlib import Path
import itertools
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import numpy as np

DIR = Path(__file__).absolute().parent
sys.path.append(str(DIR.parent))

import train


def _write(data: List[Dict[str, Any]], filename: str, **kwargs) -> bool:
    df = pd.DataFrame(data)
    for k, v in kwargs.items():
        df[k] = v
    df.to_csv(filename, index=False)
    return True


if __name__ == "__main__":
    filename = str(DIR / "synthetic.yaml")
    DAMPERS = ["adadamp", "gd", "cntsdamplr", "adagrad"]
    start, n_seeds = 0, 100
    SEEDS = list(range(start, start + n_seeds))

    DATA_DIR = DIR / "train-data"

    for seed, damper in itertools.product(SEEDS, DAMPERS):
        epochs = 70
        data, train_data = train._choose_params_and_run(
            damper=damper,
            seed=seed,
            filename=filename,
            epochs=epochs,
            max_batch_size=np.nan if damper == "adadamp" else None,
        )
        kwargs = dict(seed=seed, damper=damper)
        _write(data, str(DATA_DIR / f"{seed}-{damper}.csv"), **kwargs)
        _write(train_data, str(DATA_DIR / f"{seed}-{damper}-train.csv"), **kwargs)
