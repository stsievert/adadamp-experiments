
""""
TODO:

- [ ] write out proper train/test split. Do validation split in this script.

"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np

import train

DIR = Path(__file__).absolute().parent
DS_DIR = DIR / "dataset"
sys.path.extend([DIR.parent, DIR.parent / "train"])

class MLP(nn.Module):
    def __init__(self, in_feats=768, layer1=400, layer2=200, n_labels=70):
        super().__init__()
        self.map1 = nn.Linear(in_feats, layer1)
        self.map2 = nn.Linear(layer1, layer2)
        self.map3 = nn.Linear(layer2, n_labels)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        y_hat = F.log_softmax(self.map3(x))
        return y_hat


if __name__ == "__main__":
    with open(DS_DIR / "embedding.pt", "rb") as f:
        data = torch.load(f, weights_only=True)
    X_train, X_test, y_train, y_test = [data[k] for k in ["X_train", "X_test", "y_train", "y_test"]]

    assert tuple(X_train.shape) == (6667, 768) and tuple(X_test.shape) == (3333, 768)
    assert tuple(y_train.shape) == (6667, ) and tuple(y_test.shape) == (3333, )
    assert len(np.unique(y_train)) == 70

    est = MLP()
    params = {k: x for k, x in est.named_parameters()}
    nele = {k: x.nelement() for k, x in params.items()}
    assert sum(nele.values()) == 401_870

    #train_set = TensorDataset(X_train, y_train)
    #test_set = TensorDataset(X_test, y_test)

    train.main(
        dataset=None,
        model=est,
        epochs=20,
        damper="adagrad",
        lr=10e-2,
        train_data=(X_train, y_train),
        test_data=(X_test, y_test),
        tuning=True,
        test_freq=1,
        cuda=False,
        random_state=42,
        init_seed=42,
        verbose=True,
    )
    
