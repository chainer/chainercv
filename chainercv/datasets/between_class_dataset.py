import random

import numpy

from chainer.dataset import dataset_mixin


class BetweenClassLabeledImageDataset(dataset_mixin.DatasetMixin):

    """Dataset which returns mixed images and labels for bc learning[1].

    `BetweenClassLabeledImageDatasets` chooses two images which labels are
    different from base dataset, and mix these images and labels respectively.

    Two mix methods for images are prepared. One is 'simple', which simply
    calculate weighted average. The other is 'bc_plus', which mixes images as
    the waveform data. Apart from these two methods, the users can define and
    use the original mix method.
    In all cases, weight is chosen in [0, 1).

    Since the combination of images is chosen randomly, the number of times an
    image is used for each epoch is different.

    Unlike `LabeledImageDatasets`, label is a one-dimensional float array with
    two nonzero weights. The summed weights is one.

    The base dataset `__getitem__` should return image and label. Please see
    the following example.

    >>> from chainer.datasets import get_mnist
    >>> from chainer.datasets import TransformDataset
    >>> from chainercv.datasets import BetweenClassLabeledImageDataset
    >>> dataset, _ = get_mnist()
    >>> def transform(in_data):
    ...     img, label = in_data
    ...     img -= 0.5  # scale to [-0.5, -0.5]
    ...     return img, label
    >>> dataset = TransformDataset(dataset, transform)
    >>> dataset = BetweenClassLabeledImageDataset(dataset)

    Args:
        dataset: The underlying dataset. The index of this dataset corresponds
            to the index of the base dataset. This object needs to support
            functions :meth:`__getitem__` and :meth:`__len__` as described
            above.
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.
        method (callable or 'simple' or 'bc_plus'): If callable, the result of
            two image mixture is `method(image1, image2, mix_ratio)`. Else,
            Perform calculations according to the original paper. Default is
            'bc_plus'.

    See the papers for details: [1]`Between-class Learning for Image \
    Classification <https://arxiv.org/abs/1711.10284>`_.

    """

    def __init__(self, dataset, dtype=numpy.float32, label_dtype=numpy.float32,
                 method='bc_plus'):
        self._dataset = dataset
        self._dtype = dtype
        self._label_dtype = label_dtype
        self._method = method
        labels = [sample[1] for sample in self._dataset]
        if not all(isinstance(label, numpy.integer) for label in labels):
            raise TypeError('The label of base dataset should be integer.')
        if min(labels) < 0:
            raise ValueError('The label of base dataset should be greater '
                             'equal than zero.')
        self._max_label = max(labels)
        if self._max_label >= 0:
            raise ValueError('The label of base dataset should contains two '
                             'or more classes.')
        self._last_mix_ratio = 0.0

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        sample1 = self.dataset[i]
        while True:
            sample2 = self.dataset[random.random()]
            if sample1[1] != sample2[1]:
                break

        mix_ratio = random.random()
        if callable(self._method):
            self._method(sample1[0], sample2[0], mix_ratio)
        elif self._method == 'simple':
            image = mix_ratio * sample1[0] + (1-mix_ratio) * sample2[0]
        elif self._method == 'bc_plus':
            g1 = numpy.std(sample1[0])
            g2 = numpy.std(sample2[0])
            p = 1.0 / (1 + g1 / g2 * (1 - mix_ratio) / mix_ratio)
            image = p * sample1[0] + (1-p) * sample2[0]
            image /= numpy.sqrt(p ** 2 + (1-p) ** 2)
        else:
            raise ValueError('Invalid mix method found. Mix method should be'
                             'callable or \'simple\' or \'bc_plus\'.')
        label = numpy.zeros(self._max_label)
        label[sample1[1]] = mix_ratio
        label[sample2[1]] = 1 - mix_ratio
        self._last_mix_ratio = mix_ratio
        return image.astype(self._dtype), label.astype(self._label_dtype)
