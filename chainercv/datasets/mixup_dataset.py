import numpy

from chainer.dataset import dataset_mixin


class MixUpSoftLabelDataset(dataset_mixin.DatasetMixin):

    """Dataset which returns mixed images and labels for mixup learning[1].

    `MixUpSoftLabelDataset` chooses two images randomly and mix these images
    and labels respectively by weighted average.

    Unlike `LabeledImageDatasets`, label is a one-dimensional float array with
    two nonzero weights (soft label). The summed weights is one.

    The base dataset `__getitem__` should return image and label. Please see
    the following example.

    Example:

        We construct a mixup dataset from MNIST.

        .. code::

            >>> from chainer.datasets import get_mnist
            >>> from chainercv.datasets import SiameseDataset
            >>> from chainercv.datasets import MixUpSoftLabelDataset
            >>> mnist, _ = get_mnist()
            >>> base_dataset = SiameseDataset(mnist, mnist)
            >>> dataset = MixUpSoftLabelDataset(base_dataset)
            >>> mixed_image, mixed_label = dataset[0]

    Args:
        dataset: The underlying dataset. dataset should returns two image
            and their label. Typically, dataset is `SiameseDataset`.
            More over, each element of each dataset should have same shape.
            dataset also needs to support `__len__` as described above.
        max_label: Max label index of base datasets.

    See the papers for details: [1]`mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.

    """

    def __init__(self, dataset, max_label):
        self._dataset = dataset
        self._max_label = max_label

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        image1, label1, image2, label2 = self._dataset[i]
        mix_ratio = numpy.random.random()

        image = mix_ratio * image1 + (1-mix_ratio) * image2
        label = numpy.zeros(self._max_label + 1, dtype=numpy.float32)
        label[label1] += mix_ratio
        label[label2] += 1 - mix_ratio
        return image, label
