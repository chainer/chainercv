[![](docs/images/logo.png)](http://chainercv.readthedocs.io/en/stable/)

[![travis](https://travis-ci.org/chainer/chainercv.svg?branch=master)](https://travis-ci.org/chainer/chainercv)

# ChainerCV: a Library for Deep Learning in Computer Vision

ChainerCV is a collection of tools to train and run neural networks for computer vision tasks using [Chainer](https://github.com/chainer/chainer).

You can find the documentation [here](http://chainercv.readthedocs.io/en/stable/).

Supported tasks:

+ Object Detection ([tutorial](http://chainercv.readthedocs.io/en/latest/tutorial/detection.html), [Faster R-CNN](examples/faster_rcnn), [SSD](examples/ssd))
+ Semantic Segmentation ([SegNet](examples/segnet),)
+ Image Classification ([VGG](examples/vgg),)

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

Environments under Python 2.7.12 and 3.6.0 are tested.

+ The master branch will work on both the stable version (v2) and the development version (v3).
+ For users using Chainer v1, please use version `0.4.11`, which can be installed by `pip install chainercv==0.4.11`. This branch is unmaintained.

# Data conventions

+ Image
  + The order of color channel is RGB.
  + Shape is CHW (i.e. `(channel, height, width)`).
  + The range of values is `[0, 255]`.
  + Size is represented by row-column order (i.e. `(height, width)`).
+ Bounding Boxes
  + Shape is `(R, 4)`.
  + Coordinates are ordered as `(y_min, x_min, y_max, x_max)`. The order is the opposite of OpenCV.
+ Semantic Segmentation Image
  + Shape is `(height, weight)`. 
  + The value is class id, which is in range `[0, n_class - 1]`.

# Sample Visualization

![Example are outputs of detection models supported by ChainerCV](https://cloud.githubusercontent.com/assets/2062128/26337670/44a2a202-3fb5-11e7-8b88-6eb9886a9915.png)
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
