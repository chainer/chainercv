<!--[![travis](https://travis-ci.org/pfnet/chainercv.svg?branch=master)](https://travis-ci.org/pfnet/chainercv)-->

<!--[![pypi](https://img.shields.io/pypi/v/chainercv.svg)](https://pypi.python.org/pypi/chainercv)-->


# ChainerCV

ChainerCV is a collection of tools to train neural networks for computer vision tasks using [Chainer](https://github.com/pfnet/chainer).

You can find the documentation [here](http://chainercv.readthedocs.io/en/latest/).

This project is under development, and some API may change in the future.


# Installation

```
pip install chainercv
```


### Requirements

+ [Chainer](https://github.com/pfnet/chainer) and its dependencies
+ Pillow

For additional features

+ Matplotlib
+ OpenCV
+ Scikit-Learn


Environments under Python 2.7.12 and 3.6.0 are tested.


# Features

## Transforms

ChainerCV supports functions commonly used to prepare image data before feeding to neural networks.
We expect users to use these functions together with instantiations of `chainer.dataset.DatasetMixin`.
Many of the datasets prepared in ChainerCV are very thin wrappers around raw datasets in the filesystem, and
the transforms work best with such thin dataset classes.
The users can create a custom preprocessing pipeline by defining a function that describes
procedures to transform data.

Here is an example where the user pad images to a given shape and subtract a constant from one of the images.
This is a real example that is used to preprocess images before training a neural network for Semantic Segmentation.

```python
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.transforms import extend
from chainercv.transforms import pad

dataset = VOCSemanticSegmentationDataset()

def transform(in_data):
    # in_data is the returned values of VOCSemanticSegmentationDataset.get_example
    img, label = in_data
    img -= 122.5
    img = pad(img, max_size=(512, 512), bg_value=0)  # pad to (H, W) = (512, 512)
    label = pad(img, max_size=(512, 512), bg_value=-1)
    return img, label
extend(dataset, transform)
img, label = dataset[0]
```

As found in the example, `pad` is one of the transforms ChainerCV supports. Like other transforms, this is just a
function that takes arrays as input.
Also, `extend` is a function that decorates a dataset to transform the output of the method `get_example`.
`VOCSemanticSegmentationDataset` is a dataset class that automatically downloads and prepares PASCAL VOC data used for
the semantic segmentation task. Note that this example takes some time to download PASCAL VOC before starting.


# Automatic Download
ChainerCV supports automatic download of datasets. It uses Chainer's default download scheme for automatic download.
All data downloaded by ChainerCV is saved under a directory `$CHAINER_DATASET_ROOT/pfnet/chainercv`.

The default value of `$CHAINER_DATASET_ROOT` is `~/.chainer/dataset/`.
