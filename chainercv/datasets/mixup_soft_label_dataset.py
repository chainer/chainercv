import numpy as np

from chainer.dataset import dataset_mixin


class MixUpSoftLabelDataset(dataset_mixin.DatasetMixin):

    """Dataset which returns mixed images and labels for mixup learning [#]_.

    :class:`MixUpSoftLabelDataset` mixes two pairs of labeled images fetched
    from the base dataset.

    Unlike `LabeledImageDatasets`, label is a one-dimensional float array with
    at most two nonnegative weights (i.e. soft label). The sum of the two
    weights is one.

    Example:

        We construct a mixup dataset from MNIST.

        .. code::

            >>> from chainer.datasets import get_mnist
            >>> from chainercv.datasets import SiameseDataset
            >>> from chainercv.datasets import MixUpSoftLabelDataset
            >>> mnist, _ = get_mnist()
            >>> base_dataset = SiameseDataset(mnist, mnist)
            >>> dataset = MixUpSoftLabelDataset(base_dataset, 10)
            >>> mixed_image, mixed_label = dataset[0]
            >>> mixed_label.shape
            (10,)
            >>> mixed_label.dtype
            dtype('float32')

    Args:
        dataset: The underlying dataset. The dataset returns :obj:`img_0,
            label_0, img_1, label_1`, which is a tuple containing two pairs
            of an image and a label. Typically, dataset is `SiameseDataset`.

            The shapes of images and labels should be constant.
        n_class (int): The number of classes in the base dataset.
        alpha (float): A hyperparameter of Beta distribution.
            ``mix_ratio`` is sampled from :math:`B(\\alpha,\\alpha)`.
            The default value is 1.0 meaning that the distribution is
            equivalent to Uniform distribution with lower boundary of
            :math:`0` and upper boundary of :math:`1`.

    .. [#] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
        `mixup: Beyond Empirical Risk Minimization\
        <https://arxiv.org/abs/1710.09412>`_. arXiv 2017.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, [#mixup_1]_, [#mixup_1]_, [#mixup_1]_
        :obj:`label`, ":math:`(\#class,)`", :obj:`float32`, ":math:`[0, 1]`"

    .. [#mixup_1] Same as :obj:`dataset`.
    """

    def __init__(self, dataset, n_class, alpha=1.0):
        self._dataset = dataset
        self._n_class = n_class
        self._alpha = alpha

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        img_0, label_0, img_1, label_1 = self._dataset[i]
        mix_ratio = np.random.beta(self._alpha, self._alpha)

        image = mix_ratio * img_0 + (1-mix_ratio) * img_1
        label = np.zeros(self._n_class, dtype=np.float32)
        label[label_0] += mix_ratio
        label[label_1] += 1 - mix_ratio
        return image, label
