[![](docs/images/logo.png)](http://chainercv.readthedocs.io/en/stable/)

[![PyPI](https://img.shields.io/pypi/v/chainercv.svg)](https://pypi.python.org/pypi/chainercv)
[![License](https://img.shields.io/github/license/chainer/chainercv.svg)](https://github.com/chainer/chainercv/blob/master/LICENSE)
[![travis](https://travis-ci.org/chainer/chainercv.svg?branch=master)](https://travis-ci.org/chainer/chainercv)
[![Read the Docs](https://readthedocs.org/projects/chainercv/badge/?version=latest)](http://chainercv.readthedocs.io/en/latest/?badge=latest)

# ChainerCV: a Library for Deep Learning in Computer Vision

ChainerCV is a collection of tools to train and run neural networks for computer vision tasks using [Chainer](https://github.com/chainer/chainer).

You can find the documentation [here](http://chainercv.readthedocs.io/en/stable/).

Supported tasks:

+ Image Classification ([ResNet](examples/resnet), [SENet](examples/senet), [VGG](examples/vgg))
+ Object Detection ([tutorial](http://chainercv.readthedocs.io/en/latest/tutorial/detection.html), [Faster R-CNN](examples/faster_rcnn), [FPN](examples/fpn), [SSD](examples/ssd), [YOLO](examples/yolo), [Light-Head R-CNN](examples/light_head_rcnn))
+ Semantic Segmentation ([SegNet](examples/segnet), [PSPNet](examples/pspnet), [DeepLab v3+](examples/deeplab))
+ Instance Segmentation ([FCIS](examples/fcis), [Mask R-CNN](examples/fpn))

# Guiding Principles
ChainerCV is developed under the following three guiding principles.

+ **Ease of Use** -- Implementations of computer vision networks with a cohesive and simple interface.
+ **Reproducibility** -- Training scripts that are perfect for being used as reference implementations.
+ **Compositionality** -- Tools such as data loaders and evaluation scripts that have common API.

# Installation

```bash
$ pip install -U numpy
$ pip install chainercv
```

The instruction on installation using Anaconda is [here](http://chainercv.readthedocs.io/en/stable/#install-guide) (recommended).

### Requirements

+ [Chainer](https://github.com/chainer/chainer) and its dependencies
+ Pillow
+ Cython (Build requirements)

For additional features

+ Matplotlib
+ OpenCV
+ SciPy
+ mpi4py
+ [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)

Environments under Python 2.7.12 and 3.6.0 are tested.

+ The master branch is designed to work on Chainer v6 (the stable version) and v7 (the development version).
+ The following branches are kept for the previous version of Chainer. Note that these branches are unmaintained.
    + `0.4.11` (for Chainer v1). It can be installed by `pip install chainercv==0.4.11`.
    + `0.7` (for Chainer v2). It can be installed by `pip install chainercv==0.7`.
    + `0.8` (for Chainer v3). It can be installed by `pip install chainercv==0.8`.
    + `0.10` (for Chainer v4). It can be installed by `pip install chainercv==0.10`.
    + `0.12` (for Chainer v5). It can be installed by `pip install chainercv==0.12`.
    + `0.13` (for Chainer v6). It can be installed by `pip install chainercv==0.13`.

# Data Conventions

+ Image
  + The order of color channel is RGB.
  + Shape is CHW (i.e. `(channel, height, width)`).
  + The range of values is `[0, 255]`.
  + Size is represented by row-column order (i.e. `(height, width)`).
+ Bounding Boxes
  + Shape is `(R, 4)`.
  + Coordinates are ordered as `(y_min, x_min, y_max, x_max)`. The order is the opposite of OpenCV.
+ Semantic Segmentation Image
  + Shape is `(height, width)`.
  + The value is class id, which is in range `[0, n_class - 1]`.

# Sample Visualization

![Example are outputs of detection models supported by ChainerCV](https://user-images.githubusercontent.com/3014172/40634581-bb01f52a-6330-11e8-8502-ba3dacd81dc8.png)
These are the outputs of the detection models supported by ChainerCV.


# Citation

If ChainerCV helps your research, please cite the paper for ACM Multimedia Open Source Software Competition.
Here is a BibTeX entry:

```
@inproceedings{ChainerCV2017,
    author = {Niitani, Yusuke and Ogawa, Toru and Saito, Shunta and Saito, Masaki},
    title = {ChainerCV: a Library for Deep Learning in Computer Vision},
    booktitle = {ACM Multimedia},
    year = {2017},
}
```

The preprint can be found in arXiv: https://arxiv.org/abs/1708.08169
