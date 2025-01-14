from pathlib import Path
import sys
from time import time

import torch
import numpy as np

import train
print(train.__file__)

DIR = Path(__file__).absolute().parent
DS_DIR = DIR / "dataset"
sys.path.extend([str(DIR.parent), str(DIR.parent / "train")])

from single import MLP, Wrapper

if __name__ == "__main__":
    with open(DS_DIR / "embedding.pt", "rb") as f:
        data = torch.load(f, weights_only=True)
    X_train, X_test, y_train, y_test = [data[k] for k in ["X_train", "X_test", "y_train", "y_test"]]

    assert tuple(X_train.shape) == (6667, 768) and tuple(X_test.shape) == (3333, 768)
    assert tuple(y_train.shape) == (6667, ) and tuple(y_test.shape) == (3333, )
    assert len(np.unique(y_train)) == 70

    start = time()
    m = Wrapper(
        MLP(), X_train, y_train, X_test, y_test,
        epochs=200, verbose=True, tuning=2, write=False,
    )
    m.set_params(
        damper="padadamp",
        dwell=50,
        initial_batch_size=128,
        max_batch_size=4096,
        lr=0.7e-3,
        momentum=0.7,
        nesterov=True,
        weight_decay=1e-6,
        reduction="min",
        wait=1000,
    )
    seeds = 3 + (np.arange(3 * 2) // 2).reshape(3, 2)
    m.fit(seeds.astype(int))
