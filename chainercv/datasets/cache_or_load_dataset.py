import chainer
import h5py
import numpy as np
import os


def cache_or_load_dataset(path, dataset=None):
    """Caches a dataset if it is not already cached, or loads it otherwise

    This caches a dataset at :obj:`path` if it has not been cached yet.
    The dataset is cached by creating a hdf5 file containing data.
    This loads a dataset cached at :obj:`path` if the file previously
    created by this function already exists.

    :obj:`dataset` is either a dataset object or :obj:`None`.
    If dataset is a dataset object, dataset returned is a
    instantiation of :class:`chainer.datasets.TupleDataset` regardless
    of existance of the cache in the filesystem.
    If :obj:`dataset` is :obj:`None` and cache can not be found,
    this function returns :obj:`None`. If :obj:`dataset` is :obj:`None` and
    cache can be found, the data loaded from the cache is returned.

    .. note::

        Data has to be same length across all examples

    Args:
        path (string): A path where dataset is cached or loaded from.
        dataset: A dataset object or :obj:`None`.

    Returns:
        a dataset or :obj:`None` depending on the argument :obj:`dataset`.
        The dataset is an instantiation of
        :class:`chainer.datasets.TupleDataset`. It loads data from a hdf5
        file locating at :obj:`path`.

    """
    if os.path.exists(path):
        return _load_cached_dataset(path)

    return _cache_dataset(dataset, path)


def _load_cached_dataset(path):
    # When the driver was default, the multiprocess loading produced error
    # when multiple processes accessed the same element.
    # By making `driver` to 'core', the problem was solved.
    h5py_data = h5py.File(path, driver='core', mode='r')
    is_datum_tuple = h5py_data['is_datum_tuple'][0]
    keys = h5py_data.keys()
    keys.remove('is_datum_tuple')
    keys.sort()

    dsets = []
    for key in keys:
        dsets.append(h5py_data[key])

    if is_datum_tuple:
        dataset = chainer.datasets.TupleDataset(*dsets)
    else:
        dataset = dsets[0]
    return dataset


def _cache_dataset(dataset, path):
    if dataset is None:
        return

    datum = dataset[0]
    is_datum_tuple = True
    if not isinstance(datum, tuple):
        is_datum_tuple = False
        datum = (datum,)
    
    # When the driver was default, the multiprocess loading produced error
    # when multiple processes accessed the same element.
    # By making `driver` to 'core', the problem was solved.
    f = h5py.File(path, driver='core', mode='w')
    dset = f.create_dataset('is_datum_tuple', (1,), dtype=np.bool)
    dset[0] = is_datum_tuple

    dsets = []
    for i, value in enumerate(datum):
        dset = f.create_dataset('{}'.format(i), (len(dataset),) + value.shape, dtype=value.dtype)
        dsets.append(dset)

    for idx in range(len(dataset)):
        datum = dataset[idx]
        if not isinstance(datum, tuple):
            datum = (datum,)
        for i, val in enumerate(datum):
            dsets[i][idx] = val

    if is_datum_tuple:
        dataset = chainer.datasets.TupleDataset(*dsets)
    else:
        dataset = dsets[0]
    return dataset


if __name__ == '__main__':
    dataset, _ = chainer.datasets.get_mnist()
    path = 'foo.hdf5'

    dataset = cache_or_load_dataset(path, dataset)
