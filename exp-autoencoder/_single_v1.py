"""
Usage:
    python single.py [ident]

``ident`` might be used with this script:

    (shell 0) $ CUDA_VISIBLE_DEVICES=0 python single.py 0
    (shell 1) $ CUDA_VISIBLE_DEVICES=1 python single.py 1

"""
from copy import deepcopy
from typing import Dict, Any
from pathlib import Path
import sys
import pickle

DIR = Path(__file__).absolute().parent
sys.path.append(str(DIR.parent))

import numpy as np
import torch
import msgpack
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from scipy.stats import loguniform, uniform

# assert torch.cuda.is_available()

import train
print(train.__file__)
import adadamp


DIR = Path(__file__).absolute().parent
OUT = DIR / "data-tuning"
if not OUT.exists():
    OUT.mkdir(exist_ok=True, parents=True)

def md5(x: Any) -> str:
    from hashlib import md5 as _md5

    return _md5(pickle.dumps(x)).hexdigest()

class Model(BaseEstimator):
    def __init__(self,
        damper,
        dataset="autoencoder",
        cuda=True,
        ident=0,
        tuning=True,  # tuning hyperparameters?
        epochs=6,
        random_state=42,
        init_seed=42,
        lr=0.01,
        verbose=1,
        nesterov=True,
        dwell=1,
        momentum=0.95,
        weight_decay=1e-6,
        rho=0.9,
        initial_batch_size=32,
        max_batch_size=None,
        dampingdelay=5,
        dampingfactor=5,
        **kwargs,
    ):
        self.damper = damper

        self.dataset = dataset
        self.cuda = cuda
        self.random_state = random_state
        self.init_seed = init_seed
        self.damper = damper
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.nesterov = nesterov
        self.dwell = dwell
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.rho = rho
        self.tuning = tuning
        self.initial_batch_size=initial_batch_size
        self.max_batch_size=max_batch_size
        self.ident = ident
        self.dampingfactor = dampingfactor
        self.dampingdelay = dampingdelay
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, *args):
        params = self.get_params()
        _params_ordered = tuple(sorted((k, v) for k, v in params.items()))
        self.ident_ = md5(_params_ordered)
        for k in ["damper", "ident"]:
            params.pop(k, None)

        data_test, data_train, model, test_set = train.main(**params)
        self.stats_ = data_test[-1]

        # save model
        model = model.to(torch.device("cpu"))
        torch.save(model.state_dict(), OUT / f"model-{damper}-{self.ident_}.pt")

        # save datasets
        with open(OUT / f"stats-{damper}-{self.ident_}.pkl", "wb") as f:
            pickle.dump(
                {
                    "data_test": data_test,
                    "data_train": data_train,
                    "test_set": test_set,
                    "params": {"damper": damper, **params},
                    "ident_": self.ident_,
                    "stats_": self.stats_,
                },
                f,
            )
        return self


    def score(self, *args):
        return -1 * self.stats_["test_loss"]  # actually validation loss since self.tuning = 1

def ints(a, b):
    return list(range(a, b + 1))

def loguniform_m1(a, b, n=10_000, random_state=42):
    a, b = min(a, b), max(a, b)
    x = loguniform(a, b).rvs(size=n, random_state=random_state)
    return (1 - x).tolist()

if __name__ == "__main__":
    args = sys.argv[1:]
    print(sys.argv)
    ident = int(args[0]) if len(args) else 0

    base_search_space = {
        "lr": loguniform(1e-5, 1e-1),
        "initial_batch_size": [16, 32, 64, 96, 128, 160, 192, 224, 256],
        "max_batch_size": [None]*3 + [256, 512, 1024, 2048, 4096, 8192, 16384],
        "dwell": [1, 2, 5, 10, 20, 50],
        "weight_decay": loguniform(1e-8, 1e-4),
        "momentum": loguniform_m1(1e-1, 1e-3),
    }
    damper_search_space: Dict[str, Dict[str, Any]] = {
        "adadamp": {},
        "geodamp": {"dampingdelay": ints(1, 10), "dampingfactor": ints(1, 21)},
        "radadamp": {"rho": loguniform_m1(1e-5, 1e-1)},
        "adagrad": {"lr": loguniform(1e-3, 1e-1)},
    }

    # gd use {1384, 786, 1076}M
    for damper in [
        "sgd",
        "radadamp",  # 1391M
        "adadamp",
        "gd",  # GPU mem 1378M, 90/300W. 
        "adagrad",
        "geodamp",
        "adadelta",
    ]:
        print(f"## Training w/ {damper}")
        space = deepcopy(base_search_space)
        space.update(damper_search_space.get(damper, {}))
        space.update({"damper": [damper]})

        # about 794M per job for sgd
        m = Model(damper, epochs=100, verbose=True, tuning=1, ident=ident)
        search = RandomizedSearchCV(m, space, n_iter=256, n_jobs=28, refit=False, verbose=3, random_state=42 + int(ident), cv=[([0, 1], [2])])
        search.fit(np.random.uniform(size=(3, 2)))

        with open(OUT / f"search-{damper}-{ident}.pkl", "wb") as f:
            pickle.dump(search, f)
        #main(damper, **param)
