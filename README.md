This folder is the implementation of "Improving the convergence of SGD through
adaptive batch sizes."

For efficient sizing, we omit the training/test data used to generate each of
the figures.

## Directory structure

* `exp-*`: Directories for each figure in the paper. These directories have the
  following folders and sub-directories:
    * `train.py`: For training the models.
    * `Viz.ipynb`: To visualize the performance results on the test set
      after tuning/training. This notebook outputs figures in `figs/`.
    * `figs/`: For the figures in the paper.
* `adadamp/`: The Adadamp/PadaDamp/GeoDamp implementation (except for
  `exp-forest`, which has it's own implementation).
* `train/`: code for training the models that depends on the `adadamp` package
  being installed.

By default, these experiments will be assumed to have the Adadamp's conda environment. To activate this environment, run this code:

``` shell
$ git clone --recursive https://github.com/stsievert/adadamp-experiments.git
$ cd adadamp-experiments/adadamp
$ conda env create -f adadamp.yaml
$ pip install -e .
$ cd ..
```

The environment in `exp-cifar10` and `exp-forest` are different. See
`exp-forest/skorch.yaml` or the README in `exp-cifar10`.
