<!--[![travis](https://travis-ci.org/pfnet/chainercv.svg?branch=master)](https://travis-ci.org/pfnet/chainercv)-->

<!--[![pypi](https://img.shields.io/pypi/v/chainercv.svg)](https://pypi.python.org/pypi/chainercv)-->


# ChainerCV

ChainerCV is a collection of tools to train neural networks for computer vision tasks using [Chainer](https://github.com/pfnet/chainer).

You can find the documentation [here](http://chainercv.readthedocs.io/en/latest/).

This project is under heavy development, and API has been changing rapidly.


# Installation

```
pip install chainercv
```


### Requirements

+ [Chainer](https://github.com/pfnet/chainer) and its dependencies
+ Cython
+ Pillow

For additional features

+ Matplotlib
+ OpenCV


Environments under Python 2.7.12 and 3.6.0 are tested.


# Features

## Transforms

ChainerCV supports functions commonly used to prepare data before feeding to a neural network.
We expect users to use these functions together with an object that supports the dataset interface (e.g. `chainer.dataset.DatasetMixin`).
The users can create a custom preprocessing pipeline by defining a function that describes a
procedure to transform the incoming data.

Here is an example where the user rescales and applies a randomly rotation to an image.

```python
from chainer.datasets import get_mnist
from chainercv.datasets import TransformDataset
from chainercv.transforms import random_rotate

dataset, _ = get_mnist(ndim=3)

def transform(in_data):
    # in_data is values returned by `__getitem__` method of MNIST dataset.
    img, label = in_data
    img -= 0.5  # rescale to [-0.5, 0.5]
    img = random_rotate(img)
    return img, label
dataset = TransformDataset(dataset, transform)
img, label = dataset[0]
```

As found in the example, `random_rotate` is one of the transforms ChainerCV supports. Like other transforms, this is just a
function that takes an array as input.
Also, `TransformDataset` is a new dataset class added in ChainerCV that overrides the underlying dataset's `__getitem__` by calling `transform` as post processing.


# Automatic Download
ChainerCV supports automatic download of datasets. It uses Chainer's default download scheme for automatic download.
All data downloaded by ChainerCV is saved under a directory `$CHAINER_DATASET_ROOT/pfnet/chainercv`.

The default value of `$CHAINER_DATASET_ROOT` is `~/.chainer/dataset/`.
