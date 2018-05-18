import six

import chainer


def _as_tuple(t):
    if isinstance(t, (tuple, list)):
        return tuple(t)
    else:
        return t,


def _as_indices(keys, key_names):
    keys = _as_tuple(keys)
    key_names = _as_tuple(key_names)

    for key in keys:
        if isinstance(key, int):
            key_index = key
            if key_index < 0:
                key_index += len(key_names)
            if key_index not in range(0, len(key_names)):
                raise IndexError(
                    'index {} is out of bounds for keys with size {}'.format(
                        key, len(key_names)))
        else:
            try:
                key_index = key_names.index(key)
            except ValueError:
                raise KeyError('{} does not exists'.format(key))
        yield key_index


class SliceableDataset(chainer.dataset.DatasetMixin):
    """An abstract dataset class that supports slicing.

    This ia a dataset class that supports slicing.
    A dataset class inheriting this class should implement
    three methods: :meth:`__len__`, :meth:`keys`, and
    :meth:`get_example_by_keys`.

    Users can easily create sliceable datasets using
    :class:`~chainercv.chainer_experimental.datasets.sliceable.GetterDataset`
    or
    :class:`~chainercv.chainer_experimental.datasets.sliceable.TupleDataset`.
    """

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        """Return names of all keys

        Returns:
            string or tuple of strings
        """
        raise NotImplementedError

    def get_example_by_keys(self, index, key_indices):
        """Return data of an example by keys

        Args:
            index (int): An index of an example.
            key_indices (tuple of ints): A tuple of indices of requested keys.

        Returns:
            tuple of data
        """
        raise NotImplementedError

    def get_example(self, index):
        if isinstance(self.keys, tuple):
            return self.get_example_by_keys(
                index, tuple(range(len(self.keys))))
        else:
            return self.get_example_by_keys(index, (0,))[0]

    @property
    def slice(self):
        return SliceHelper(self)

    def __iter__(self):
        return (self.get_example(i) for i in six.moves.range(len(self)))


class SliceHelper(object):
    """A helper class for :class:`SliceableDataset`."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, args):
        if isinstance(args, tuple):
            index, keys = args
        else:
            index = args
            keys = self._dataset.keys

        key_indices = tuple(_as_indices(keys, self._dataset.keys))
        return_tuple = isinstance(keys, (list, tuple))

        return SlicedDataset(
            self._dataset, index,
            tuple(key_indices) if return_tuple else key_indices[0])


class SlicedDataset(SliceableDataset):
    """A sliced view for :class:`SliceableDataset`."""

    def __init__(self, dataset, index, key_indices):
        self._dataset = dataset
        self._index = index
        self._key_indices = key_indices

    def __len__(self):
        if isinstance(self._index, slice):
            start, end, step = self._index.indices(len(self._dataset))
            return len(range(start, end, step))
        else:
            return len(self._index)

    @property
    def keys(self):
        keys = _as_tuple(self._dataset.keys)
        if isinstance(self._key_indices, tuple):
            return tuple(keys[key_index] for key_index in self._key_indices)
        else:
            return keys[self._key_indices]

    def get_example_by_keys(self, index, key_indices):
        if isinstance(key_indices, tuple):
            key_indices = tuple(
                _as_tuple(self._key_indices)[key_index]
                for key_index in key_indices)
        else:
            key_indices = _as_tuple(self._key_indices)[key_indices]

        if isinstance(self._index, slice):
            start, _, step = self._index.indices(len(self._dataset))
            return self._dataset.get_example_by_keys(
                start + index * step, key_indices)
        else:
            return self._dataset.get_example_by_keys(
                self._index[index], key_indices)
