from chainercv.chainer_experimental.datasets.sliceable.sliceable_dataset \
    import _as_key_indices
from chainercv.chainer_experimental.datasets.sliceable.sliceable_dataset \
    import _is_iterable
from chainercv.chainer_experimental.datasets.sliceable import SliceableDataset


class GetterDataset(SliceableDataset):
    """A sliceable dataset class that is defined with getters.

    This is a dataset class with getters.
    Please refer to the tutorial for more detailed explanation.

    Here is an example.

    >>> class SliceableLabeledImageDataset(GetterDataset):
    >>>     def __init__(self, pairs, root='.'):
    >>>         super(SliceableLabeledImageDataset, self).__init__()
    >>>         with open(pairs) as f:
    >>>             self._pairs = [l.split() for l in f]
    >>>         self._root = root
    >>>
    >>>         self.add_getter('img', self.get_image)
    >>>         self.add_getter('label', self.get_label)
    >>>
    >>>     def __len__(self):
    >>>         return len(self._pairs)
    >>>
    >>>     def get_image(self, i):
    >>>         path, _ = self._pairs[i]
    >>>         return read_image(os.path.join(self._root, path))
    >>>
    >>>     def get_label(self, i):
    >>>         _, label = self._pairs[i]
    >>>         return np.int32(label)
    >>>
    >>> dataset = SliceableLabeledImageDataset('list.txt')
    >>>
    >>> # get a subset with label = 0, 1, 2
    >>> # no images are loaded
    >>> indices = [i for i, label in
    ...            enumerate(dataset.slice[:, 'label']) if label in {0, 1, 2}]
    >>> dataset_012 = dataset.slice[indices]
    """

    def __init__(self):
        self._keys = []
        self._getters = []
        self._return_tuple = True

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        if self._return_tuple:
            return tuple(key for key, _, _ in self._keys)
        else:
            return self._keys[0][0]

    @keys.setter
    def keys(self, keys):
        self._keys = [self._keys[key_index]
                      for key_index in _as_key_indices(keys, self.keys)]
        self._return_tuple = _is_iterable(keys)

    def add_getter(self, keys, getter):
        """Register a getter function

        Args:
            keys (int or string or tuple of strings): The number or name(s) of
                data that the getter function returns.
            getter (callable): A getter function that takes an index and
                returns data of the corresponding example.
        """
        self._getters.append(getter)
        if isinstance(keys, int):
            if keys == 1:
                keys = None
            else:
                keys = (None,) * keys
        if _is_iterable(keys):
            for key_index, key in enumerate(keys):
                self._keys.append((key, len(self._getters) - 1, key_index))
        else:
            self._keys.append((keys, len(self._getters) - 1, None))

    def get_example_by_keys(self, index, key_indices):
        example = []
        cache = {}
        for key_index in key_indices:
            _, getter_index, key_index = self._keys[key_index]
            if getter_index not in cache:
                cache[getter_index] = self._getters[getter_index](index)
            if key_index is None:
                example.append(cache[getter_index])
            else:
                example.append(cache[getter_index][key_index])
        return tuple(example)
