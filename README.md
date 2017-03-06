[![travis](https://travis-ci.org/yuyu2172/chainercv.svg?branch=master)](https://travis-ci.org/yuyu2172/chainercv)

<!--[![pypi](https://img.shields.io/pypi/v/chainercv.svg)](https://pypi.python.org/pypi/chainercv)-->

# ChainerCV

ChainerCV does the dirty work when training a neural network for a computer vision task. In particular, this is a colletion of tools that do following

1. Thin wrappers around common computer vision datasets with support for automatic downloads
2. Composition of data preprocessing (e.g. data augmentation after padding of data)
3. Trainer extensions for computer vision tasks (e.g. visualization of outputs for semantic segmentation)
4. Evaluation metrics for various computer vision tasks


You can find the documentation [here](http://chainercv.readthedocs.io/en/latest/).



# Installation

```
pip install chainercv
```


### Requirements

+ [Chainer](https://github.com/pfnet/chainer) and its dependencies
+ Pillow
+ Matplotlib

For additional features

+ Scikit-Image


# Features

## Transforms

ChainerCV supports functions commonly used to prepare image data before feeding to neural networks.
We expect users to use these functions together with instantiations of `chainer.dataset.DatasetMixin`.
Many of the datasets prepared in ChainerCV are very thin wrappers around raw datasets in the filesystem, and
the transforms work best with such thin dataset classes.
The users can create a custom preprocessing pipeline by defining a function that describes
procedures to transform data using full functionality of Python language.

Here is an example where the user pad images to the given shape and subtract a constant from one of the images.
This is a real example that is used to preprocess images before training a neural network for Semantic Segmentation.

```python
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.transforms import extend
from chainercv.transforms import random_crop

dataset = VOCSemanticSegmentationDataset()

def transform(in_data):
    # in_data is the returned values of VOCSemanticSegmentationDataset.get_example
    img, label = in_data
    img, label = random_crop((img, label), (None, 256, 256))  # pad to (H, W) = (256, 256)
    img -= 122.5
    return img, label
extend(dataset, transform)
img, label = dataset[0]
```

As found in the example, `random_crop` is one of the transforms ChainerCV supports. Like other transforms, this is just a
function that takes arrays as input.
Also, `extend` is a function that decorates a dataset to transform the output of the method `get_example`.
`VOCSemanticSegmentationDataset` is a dataset class that automatically downloads and prepares PASCAL VOC data used for
semantic segmentation tasks. Note that this example takes some time to download PASCAL VOC before starting.


# Automatic Download
ChainerCV supports automatic download of datasets. It uses Chainer's default download scheme for automatic download.
Therefore, the downloaded data is stored under a directory `$CHAINER_DATASET_ROOT/pfnet/chainercv`.

The default value of `$CHAINER_DATASET_ROOT` is `~/.chainer/dataset/`.
If you want to change the directory where you download and look up datasets, please change the value of the global value `$CHAINER_DATASET_ROOT` by a command like below.

```
export CHAINER_DATASET_ROOT=/CLOUD/dataset/  # this is an exmaple
``` 
