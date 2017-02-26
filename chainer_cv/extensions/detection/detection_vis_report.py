import collections
import numpy as np
import os.path as osp
from skimage.color import label2rgb
import warnings

import chainer
from chainer.utils import type_check

from chainer_cv.extensions.utils import check_type
from chainer_cv.extensions.utils import forward

try:
    from matplotlib import pyplot as plt
    _available = True

except ImportError:
    _available = False


def _check_available():
    if not _available:
        warnings.warn('matplotlib is not installed on your environment, '
                      'so nothing will be plotted at this time. '
                      'Please install matplotlib to plot figures.\n\n'
                      '  $ pip install matplotlib\n')


class DetectionVisReport(chainer.training.extension.Extension):
    """An extension that visualizes output of a detection model.


