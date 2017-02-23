import numpy as np

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


def keep_subset(func, indices):

    def wrapped(i):
        wrapped_i = indices[i]
        return func(wrapped_i)
    return wrapped


class KeepSubsetWrapper(DatasetWrapper):

    """Keep subset of dataset.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        indices (list of int): inices of the examples that are used after
            wrapping the dataset. The order of examples from the wrapped
            dataset corresponds to the ordering of ints in `indices`.
        wrapped_func_names (list of strings): List of the name of functions
            which you will be affected by selecting subset of all indices.
            These functions include `get_example` and similar functions that
            take (self, i) as its argument.
            If `wrapped_func_names` is `None`, this arguement will be
            `[\'get_example\']`.

    """

    def __init__(self, dataset, indices, wrapped_func_names=None):
        super(KeepSubsetWrapper, self).__init__(dataset)

        self._indices = indices
        if (np.max(self._indices) > len(self._dataset) or
                np.min(self._indices) < 0):
            raise ValueError('indices need to be a valid index to a dataset')

        if wrapped_func_names is None:
            wrapped_func_names = ['get_example']

        for name in wrapped_func_names:
            wrapped_func = keep_subset(
                getattr(self, name), self._indices)
            setattr(self, name, wrapped_func)

    def __len__(self):
        return len(self._indices)
