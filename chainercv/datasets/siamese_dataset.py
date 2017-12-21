import numpy as np

import chainer


class SiameseDataset(chainer.dataset.DatasetMixin):

    """A dataset that returns samples fetched from two datasets.

    The dataset returns samples from the two base datasets.
    If :obj:`pos_ratio` is not :obj:`None`,
    :class:`SiameseDataset` can be configured to return positive
    pairs at the ratio of :obj:`pos_ratio` and negative pairs at the ratio
    of :obj:`1 - pos_ratio`.
    In this mode, the base datasets are assumed to be label datasets that
    return an image and a label as a sample.
    Note that if :class:`SiameseDataset` cannot find a pair with the desired
    status (pos/neg), it fallbacks on fetching an arbitrary pair.
    This means that even if :obj:`pos_ratio=1`, this dataset will return a
    negative pair if it is impossible to construct a positive pair.

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

    """

    def __init__(self, dataset_0, dataset_1,
                 pos_ratio=None, labels_0=None, labels_1=None):
        self._dataset_0 = dataset_0
        self._dataset_1 = dataset_1
        self._pos_ratio = pos_ratio

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
        self._labels_0 = labels_0
        self._labels_1 = labels_1

    def __len__(self):
        return len(self._dataset_0)

    def get_example(self, i):
        if self._pos_ratio is None:
            idx0 = np.random.choice(np.arange(len(self._dataset_0)))
            idx1 = np.random.choice(np.arange(len(self._dataset_1)))
        else:
            n_search = 3
            for count in range(n_search):
                idx0 = np.random.choice(np.arange(len(self._dataset_0)))
                label_0 = self._labels_0[idx0]
                if np.random.binomial(1, self._pos_ratio):
                    # get pos-pair
                    idx1_candidates = np.where(self._labels_1 == label_0)[0]
                else:
                    # get neg-pair
                    idx1_candidates = np.where(self._labels_1 != label_0)[0]
                if len(idx1_candidates) > 0:
                    break
                elif count == n_search - 1:
                    idx1_candidates = np.arange(len(self._labels_1))
            idx1 = np.random.choice(idx1_candidates)

        example_0 = self._dataset_0[idx0]
        example_1 = self._dataset_1[idx1]
        return tuple(example_0) + tuple(example_1)
