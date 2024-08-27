from pathlib import Path
import sys
import pickle

DIR = Path(__file__).absolute().parent
sys.path.append(str(DIR.parent))

import torch

assert torch.cuda.is_available()

import train
import adadamp


def main(damper, **kwargs):
    params = dict(
        dataset="autoencoder",
        cuda=True,
        random_state=42,
        init_seed=42,
        damper=damper,
        epochs=1,
        lr=0.01,
        verbose=0.2,
        nesterov=True,
        momentum=0.95,
        weight_decay=1e-6,
        rho=0.9,
    )
    params.update(kwargs)

    data_test, data_train, model, test_set = train.main(**params)
    model = model.to(torch.device("cpu"))
    torch.save(model.state_dict(), f"temp-{damper}-model.pt")
    with open(f"temp-{damper}.pkl", "wb") as f:
        pickle.dump(
            {
                "data_test": data_test,
                "data_train": data_train,
                "test_set": test_set,
            },
            f,
        )


if __name__ == "__main__":
    for damper in ["adadamp", "adadelta", "adagrad", "sgd", "geodamp", "gd"]:
        print(f"Training w/ {damper}")
        main(damper)
