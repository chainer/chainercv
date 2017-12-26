import numpy as np

import chainer


def _construct_label_to_key(labels):
    d = dict()
    for i, l in enumerate(labels):
        if l not in d:
            d[l] = list()
        d[l].append(i)
    return d


class SiameseDataset(chainer.dataset.DatasetMixin):

    """A dataset that returns samples fetched from two datasets.

    The dataset returns samples from the two base datasets.
    If :obj:`pos_ratio` is not :obj:`None`,
    :class:`SiameseDataset` can be configured to return positive
    pairs at the ratio of :obj:`pos_ratio` and negative pairs at the ratio
    of :obj:`1 - pos_ratio`.
    In this mode, the base datasets are assumed to be label datasets that
    return an image and a label as a sample.

    Args:
        dataset_0: The first base dataset.
        dataset_1: The second base dataset.
        pos_ratio (None or float): If this is not :obj:`None`,
            this dataset tries to construct positive pairs at the
            given rate. If :obj:`None`,
            this dataset randomly samples examples from the base
            datasets. The default value is :obj:`None`.
        labels_0 (None or numpy.ndarray): The labels associated to
            the first base dataset. The length should be the same as
            the length of the first dataset. If this is :obj:`None`,
            the labels are automatically fetched using the following
            line of code: :obj:`[ex[1] for ex in dataset_0]`.
            Also, if :obj:`pos_ratio` is :obj:`None`, this value
            is ignored. The default value is :obj:`None`.
        labels_1 (None or numpy.ndarray): The labels associated to
            the second base dataset. Please consult the explanation for
            :obj:`labels_0`.
        length (None): The length of this dataset. If :obj:`None`,
            the length of the first base dataset is the length of this
            dataset.

    """

    def __init__(self, dataset_0, dataset_1,
                 pos_ratio=None, labels_0=None, labels_1=None, length=None):
        self._dataset_0 = dataset_0
        self._dataset_1 = dataset_1
        self._pos_ratio = pos_ratio
        if length is None:
            length = len(self._dataset_0)
        self._length = length

        if pos_ratio is not None:
            if labels_0 is None:
                labels_0 = [example[1] for example in dataset_0]
            if labels_1 is None:
                labels_1 = [example[1] for example in dataset_1]
            labels_0 = np.array(labels_0)
            labels_1 = np.array(labels_1)
            if not (labels_0.dtype == np.int32 and labels_0.ndim == 1
                    and len(labels_0) == len(dataset_0) and
                    labels_1.dtype == np.int32 and labels_1.ndim == 1
                    and len(labels_1) == len(dataset_1)):
                raise ValueError('the labels are invalid.')

            # label -> idx
            self._label_to_index_0 = _construct_label_to_key(labels_0)
            self._label_to_index_1 = _construct_label_to_key(labels_1)
            # construct array of labels with positive pairs
            unique_0 = np.array(self._label_to_index_0.keys())
            self._exist_pos_pair_labels_0 =\
                np.array([l for l in unique_0 if np.any(labels_1 == l)])
            if len(self._exist_pos_pair_labels_0) == 0 and pos_ratio > 0:
                raise ValueError(
                    'There is no positive pairs. For the given pair of '
                    'datasets, please set pos_ratio to None.')
            # const array of labels in dataset_0 with negative pairs
            self._exist_neg_pair_labels_0 = \
                np.array([l for l in unique_0 if np.any(labels_1 != l)])
            if len(self._exist_neg_pair_labels_0) == 0 and pos_ratio < 1:
                raise ValueError(
                    'There is no negative pairs. For the given pair of '
                    'datasets, please set pos_ratio to None.')

        self._labels_0 = labels_0
        self._labels_1 = labels_1

    def __len__(self):
        return self._length

    def get_example(self, i):
        if self._pos_ratio is None:
            idx0 = np.random.choice(np.arange(len(self._dataset_0)))
            idx1 = np.random.choice(np.arange(len(self._dataset_1)))
        else:
            # get pos-pair
            if np.random.binomial(1, self._pos_ratio):
                l = np.random.choice(self._exist_pos_pair_labels_0)
                idx0 = np.random.choice(self._label_to_index_0[l])
                idx1 = np.random.choice(self._label_to_index_1[l])
            # get neg-pair
            else:
                l0 = np.random.choice(self._exist_neg_pair_labels_0)
                keys = list(self._label_to_index_1.keys())
                if l0 in keys:
                    keys.remove(l0)
                l1 = np.random.choice(keys)
                idx0 = np.random.choice(self._label_to_index_0[l0])
                idx1 = np.random.choice(self._label_to_index_1[l1])

        example_0 = self._dataset_0[idx0]
        example_1 = self._dataset_1[idx1]
        return tuple(example_0) + tuple(example_1)
