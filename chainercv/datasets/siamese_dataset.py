import collections
import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset


def _construct_label_to_key(labels):
    d = collections.defaultdict(list)
    for i, label in enumerate(labels):
        d[label].append(i)
    return d


class SiameseDataset(GetterDataset):

    """A dataset that returns samples fetched from two datasets.

    The dataset returns samples from the two base datasets.
    If :obj:`pos_ratio` is not :obj:`None`,
    :class:`SiameseDataset` can be configured to return positive
    pairs at the ratio of :obj:`pos_ratio` and negative pairs at the ratio
    of :obj:`1 - pos_ratio`.
    In this mode, the base datasets are assumed to be label datasets that
    return an image and a label as a sample.

    Example:

        We construct a siamese dataset from MNIST.

        .. code::

            >>> from chainer.datasets import get_mnist
            >>> from chainercv.datasets import SiameseDataset
            >>> mnist, _ = get_mnist()
            >>> dataset = SiameseDataset(mnist, mnist, pos_ratio=0.3)
            # The probability of the two samples having the same label
            # is 0.3 as specified by pos_ratio.
            >>> img_0, label_0, img_1, label_1 = dataset[0]
            # The returned examples may change in the next
            # call even if the index is the same as before
            # because SiameseDataset picks examples randomly
            # (e.g., img_0_new may differ from img_0).
            >>> img_0_new, label_0_new, img_1_new, label_1_new = dataset[0]


    Args:
        dataset_0: The first base dataset.
        dataset_1: The second base dataset.
        pos_ratio (float): If this is not :obj:`None`,
            this dataset tries to construct positive pairs at the
            given rate. If :obj:`None`,
            this dataset randomly samples examples from the base
            datasets. The default value is :obj:`None`.
        length (int): The length of this dataset. If :obj:`None`,
            the length of the first base dataset is the length of this
            dataset.
        labels_0 (numpy.ndarray): The labels associated to
            the first base dataset. The length should be the same as
            the length of the first dataset. If this is :obj:`None`,
            the labels are automatically fetched using the following
            line of code: :obj:`[ex[1] for ex in dataset_0]`.
            By setting :obj:`labels_0` and skipping the fetching
            iteration, the computation cost can be reduced.
            Also, if :obj:`pos_ratio` is :obj:`None`, this value
            is ignored. The default value is :obj:`None`.
            If :obj:`labels_1` is spcified and
            :obj:`dataset_0` and :obj:`dataset_1` are the same,
            :obj:`labels_0` can be skipped.
        labels_1 (numpy.ndarray): The labels associated to
            the second base dataset. If :obj:`labels_0` is spcified and
            :obj:`dataset_0` and :obj:`dataset_1` are the same,
            :obj:`labels_1` can be skipped.
            Please consult the explanation for :obj:`labels_0`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img_0`, [#siamese_1]_, [#siamese_1]_, [#siamese_1]_
        :obj:`label_0`, scalar, :obj:`int32`, ":math:`[0, \#class - 1]`"
        :obj:`img_1`, [#siamese_2]_, [#siamese_2]_, [#siamese_2]_
        :obj:`label_1`, scalar, :obj:`int32`, ":math:`[0, \#class - 1]`"

    .. [#siamese_1] Same as :obj:`dataset_0`.
    .. [#siamese_2] Same as :obj:`dataset_1`.
    """

    def __init__(self, dataset_0, dataset_1,
                 pos_ratio=None, length=None, labels_0=None, labels_1=None):
        super(SiameseDataset, self).__init__()

        self._dataset_0 = dataset_0
        self._dataset_1 = dataset_1
        self._pos_ratio = pos_ratio
        if length is None:
            length = len(self._dataset_0)
        self._length = length

        if pos_ratio is not None:
            # handle cases when labels_0 and labels_1 are not set
            if dataset_0 is dataset_1:
                if labels_0 is None and labels_1 is None:
                    labels_0 = np.array([example[1] for example in dataset_0])
                    labels_1 = labels_0
                elif labels_0 is None:
                    labels_0 = labels_1
                elif labels_1 is None:
                    labels_1 = labels_0
            else:
                if labels_0 is None:
                    labels_0 = np.array([example[1] for example in dataset_0])
                if labels_1 is None:
                    labels_1 = np.array([example[1] for example in dataset_1])

            if not (labels_0.dtype == np.int32 and labels_0.ndim == 1 and
                    len(labels_0) == len(dataset_0) and
                    labels_1.dtype == np.int32 and labels_1.ndim == 1 and
                    len(labels_1) == len(dataset_1)):
                raise ValueError('the labels are invalid.')

            # Construct mapping label->idx
            self._label_to_index_0 = _construct_label_to_key(labels_0)
            if dataset_0 is dataset_1:
                self._label_to_index_1 = self._label_to_index_0
            else:
                self._label_to_index_1 = _construct_label_to_key(labels_1)
            # select labels with positive pairs
            unique_0 = np.array(list(self._label_to_index_0.keys()))
            self._exist_pos_pair_labels_0 =\
                np.array([l for l in unique_0 if np.any(labels_1 == l)])
            if len(self._exist_pos_pair_labels_0) == 0 and pos_ratio > 0:
                raise ValueError(
                    'There is no positive pairs. For the given pair of '
                    'datasets, please set pos_ratio to None.')
            # select labels in dataset_0 with negative pairs
            self._exist_neg_pair_labels_0 = \
                np.array([l for l in unique_0 if np.any(labels_1 != l)])
            if len(self._exist_neg_pair_labels_0) == 0 and pos_ratio < 1:
                raise ValueError(
                    'There is no negative pairs. For the given pair of '
                    'datasets, please set pos_ratio to None.')

        self._labels_0 = labels_0
        self._labels_1 = labels_1

        self.add_getter(
            ('img_0', 'label_0', 'img_1', 'label_1'), self._get_example)

    def __len__(self):
        return self._length

    def _get_example(self, i):
        if self._pos_ratio is None:
            idx0 = np.random.choice(np.arange(len(self._dataset_0)))
            idx1 = np.random.choice(np.arange(len(self._dataset_1)))
        else:
            # get pos-pair
            if np.random.binomial(1, self._pos_ratio):
                label = np.random.choice(self._exist_pos_pair_labels_0)
                idx0 = np.random.choice(self._label_to_index_0[label])
                idx1 = np.random.choice(self._label_to_index_1[label])
            # get neg-pair
            else:
                label_0 = np.random.choice(self._exist_neg_pair_labels_0)
                keys = list(self._label_to_index_1.keys())
                if label_0 in keys:
                    keys.remove(label_0)
                label_1 = np.random.choice(keys)
                idx0 = np.random.choice(self._label_to_index_0[label_0])
                idx1 = np.random.choice(self._label_to_index_1[label_1])

        example_0 = self._dataset_0[idx0]
        example_1 = self._dataset_1[idx1]
        return tuple(example_0) + tuple(example_1)
