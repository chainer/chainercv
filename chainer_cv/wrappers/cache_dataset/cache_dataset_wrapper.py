import os.path as osp
import shelve
import tempfile

from chainer_cv.wrappers.dataset_wrapper import DatasetWrapper


class CacheDatasetWrapper(DatasetWrapper):
    """This caches outputs from wrapped dataset and reuse them.

    Note that it converts outputs from wrapped dataset into numpy.ndarray.
    Unlike `CacheArrayDatasetWrapper`, this works even in the case when
    the wrapped dataset returns arrays whose shapes are not same.

    Args:
        dataset: a dataset or a wrapper that this wraps.
        copy (bool): If `copy` is True, this self.get_example returns a copy of
            cached array.

    """

    def __init__(self, dataset, copy=True):
        super(CacheDatasetWrapper, self).__init__(dataset)
        self.cache = None
        self._initialize()

    def _initialize(self):
        filename = osp.join(tempfile.mkdtemp(), 'cache.db')
        self.cache = shelve.open(filename, protocol=2)

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
        key = str(i)
        if key not in self.cache:
            self.cache[key] = self._dataset[i]
        return self.cache[key]

    def __del__(self):
        if self.cache is not None:
            self.cache.close()


if __name__ == '__main__':
    import chainer
    train, test = chainer.datasets.get_mnist()
    cached_dataset = CacheDatasetWrapper(train)
