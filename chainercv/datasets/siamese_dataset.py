import numpy as np

import chainer


class SiameseDataset(chainer.dataset.DatasetMixin):

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
            if not (labels_0.dtype == np.int32 and labels_0.ndim == 1 and
                    labels_1.dtype == np.int32 and labels_1.ndim == 1):
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
