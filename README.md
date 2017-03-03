[![travis](https://travis-ci.org/yuyu2172/chainercv.svg?branch=master)](https://travis-ci.org/yuyu2172/chainercv)

<!--[![pypi](https://img.shields.io/pypi/v/chainercv.svg)](https://pypi.python.org/pypi/chainercv)-->

# ChainerCV

ChainerCV does the dirty work when training a neural network for a computer vision task. In particular, this is a colletion of tools that do following

1. Thin wrappers around common computer vision datasets with support for automatic downloads
2. Composition of data preprocessing (e.g. data augmentation after padding of data)
3. Trainer extensions for computer vision tasks (e.g. visualization of outputs for semantic segmentation)
4. Evaluation metrics for various computer vision tasks



# Installation


```
pip install chainer-cv
```


# Automatic Download
ChainerCV supports automatic download of datasets. It uses Chainer's default download scheme for automatic download.
Therefore, the downloaded data is stored under a directory `$CHAINER_DATASET_ROOT/yuyu2172/chainercv`.

The default value of `$CHAINER_DATASET_ROOT` is `~/.chainer/dataset/`.
If you want to change the directory where you download and look up datasets, please change the value of the global value `$CHAINER_DATASET_ROOT` by a command like below.

```
export CHAINER_DATASET_ROOT=/CLOUD/dataset/  # this is an exmaple
```


# Versions
**0.2.1 (February 23rd, 2017)** The data is now saved under `$CHAINER_DATASET_ROOT/yuyu2172/chainer-cv`.

**0.2 (February 22nd, 2017)** TensorBoard visualization of embedded space [#4](https://github.com/yuyu2172/chainercv/pull/4).

**0.1 (February 20th, 2017)** Semantic Segmentation example. Initial release.
