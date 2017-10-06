[![travis](https://travis-ci.org/chainer/chainercv.svg?branch=master)](https://travis-ci.org/chainer/chainercv)

<!--[![pypi](https://img.shields.io/pypi/v/chainercv.svg)](https://pypi.python.org/pypi/chainercv)-->


# ChainerCV

ChainerCV is a collection of tools to train and run neural networks for computer vision tasks using [Chainer](https://github.com/chainer/chainer).

You can find the documentation [here](http://chainercv.readthedocs.io/en/stable/).

![Example are outputs of detection models supported by ChainerCV](https://cloud.githubusercontent.com/assets/2062128/26337670/44a2a202-3fb5-11e7-8b88-6eb9886a9915.png)
These are the outputs of the detection models supported by ChainerCV.


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


# Features

## Data conventions

ChainerCV follows the following conventions.

+ Channels of a color image are ordered as RGB. The shape of an image-array is CHW (channel, height, width).
+ Axis-dependent values follow row-column order.
  + Shape of an image is always represented as `(height, width)`.
  + Coordinates of bounding boxes are ordered as `(y_min, x_min, y_max, x_max)`. `(y_min, x_min)` and `(y_max, x_max)` are coordinates of the top left and the bottom right vertices of a bounding box respectively.
  + Coordinates of keypoints are ordered as `(y, x)`.

## Models
Currently, ChainerCV supports networks for object detection and semantic segmentation.
We are planning to support networks with different tasks in the future.
Image detection is the task of finding objects in an image and classifying the objects.
Semantic segmentation is the task of segmenting an image into pieces and assigning object labels to them.
Our implementations include:

Detection Models

+ **Faster R-CNN**
+ **Single Shot Multibox Detector (SSD)**

Semantic Segmentation 
+ **SegNet**

Models for certain tasks are designed to have common interfaces.
For example, detection models support method that takes images and outputs coordinates, class labels and confidence scores of bounding boxes predicted around estimated regions of objects.
The common interface allows users to swap different models easily inside their code.
On top of that, the utilitiy codes build on top of this interface.
For example, there is an extension of `chainer.training.Trainer` called `chainercv.extensions.DetectionVisReport` that visualizes outputs during training for models that previously stated common interface.

ChainerCV internally downloads and stores files in a file system using Chainer's downloading mechanism to conveniently download pretrained models from the internet automatically.
The convenient interface coupled with this functionality allows users to execute algorithms in two lines of code:

```python
from chainercv.links import FasterRCNNVGG16, SSD300
# You can use Faster R-CNN instead of SSD.
# model = FasterRCNNVGG16(pretrained_model='voc07')
model = SSD300(pretrained_model='voc0712')


# `bboxes` is a list of numpy arrays containing coordinates of boundnig boxes
# around objects. `labels` and `scores` are class ids and confidence scores for
# the boxes.
bboxes, labels, scores = model.predict(imgs)  # imgs is a list of image
```

With the following commands, you can run a visualization demo.

```
$ cd examples/ssd  # or cd examples/faster_rcnn
$ wget https://cloud.githubusercontent.com/assets/2062128/26187667/9cb236da-3bd5-11e7-8bcf-7dbd4302e2dc.jpg -O sample.jpg
$ python demo.py sample.jpg
```

With the following commands, the visualizations on the top of this page can be produced.

```
$ cd examples/detection
$ python visualize_models.py
```


## Transforms

ChainerCV provides functions commonly used to prepare data before feeding to a neural network.
We expect users to use these functions together with an object that supports the dataset interface (e.g. `chainer.dataset.DatasetMixin`).
Users can create a custom preprocessing pipeline by defining a function that describes a
procedure to transform the incoming data.
By decoupling preprocessing steps from dataset objects, dataset objects can be reused in a variety of preprocessing pipelines.

Here is an example where a user rescales and applies a random rotation to an image as a preprocessing step.

```python
from chainer.datasets import get_mnist
from chainer.datasets import TransformDataset
from chainercv.transforms import random_rotate

dataset, _ = get_mnist(ndim=3)

def transform(in_data):
    # in_data is values returned by __getitem__ method of MNIST dataset.
    img, label = in_data
    img -= 0.5  # rescale to [-0.5, 0.5]
    img = random_rotate(img)
    return img, label
dataset = TransformDataset(dataset, transform)
img, label = dataset[0]
```

As found in the example, `random_rotate` is one of the transforms provided by ChainerCV. Like other transforms, this is just a
function that takes an array as input.


# Citation

If ChainerCV helps your research, please cite the paper for ACM Multimedia Open Source competition.
Here is a BibTex entry:

```
@inproceedings{ChainerCV2017,
    author = {Niitani, Yusuke and Ogawa, Toru and Saito, Shunta and Saito, Masaki},
    title = {ChainerCV: a Library for Deep Learning in Computer Vision},
    booktitle = {ACM Multimedia},
    year = {2017},
}
```

The preprint can be found in arXiv: https://arxiv.org/abs/1708.08169
