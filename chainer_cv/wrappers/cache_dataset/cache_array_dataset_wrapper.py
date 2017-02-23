import numpy as np
import os.path as osp
import tempfile

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class CacheArrayDatasetWrapper(DatasetWrapper):
    """This caches outputs from wrapped dataset and reuse them.

    Note that it converts outputs from wrapped dataset into numpy.ndarray.
    This only works when the wrapped dataset returns arrays whose shapes are
    same.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        copy (bool): If `copy` is True, this self.get_example returns a copy of
            cached array.

    """

    def __init__(self, dataset, copy=True):
        super(CacheArrayDatasetWrapper, self).__init__(dataset)
        self.initialized = False
        self.cache = None
        self.has_cache = [False] * len(self._dataset)
        self.copy = copy

    def get_example(self, i):
        """Returns the i-th example.

        This caches the requested example if it has not been already cached.
        Once cached, the __getitem__ method of the wrapped dataset will not be
        called. Instead, the cached data will be loaded.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example

        """
        if not self.initialized:
            self._initialize(i)
            self.initialized = True

        if not self.has_cache[i]:
            arrays = self._dataset[i]
            for arr_i, a in enumerate(arrays):
                self.cache[arr_i][i] = np.array(a)
            self.has_cache[i] = True
        if self.copy:
            out = tuple([a_cache[i].copy() for a_cache in self.cache])
        else:
            out = tuple([a_cache[i] for a_cache in self.cache])
        return out

    def _initialize(self, i):
        arrays = self._dataset[i]
        self.cache = [None] * len(arrays)
        for arr_i, a in enumerate(arrays):
            if not isinstance(np.array(a), np.ndarray):
                raise ValueError(
                    'The dataset wrapped by CacheDatasetWrapper needs to '
                    'return tuple of numpy.ndarray')
            shape = (len(self),) + a.shape
            filename = osp.join(
                tempfile.mkdtemp(), 'cache_{}.data'.format(arr_i))
            self.cache[arr_i] = np.memmap(
                filename, dtype=a.dtype, mode='w+', shape=shape)


if __name__ == '__main__':
    import chainer
    train, test = chainer.datasets.get_mnist()
    cached_dataset = CacheArrayDatasetWrapper(train)
