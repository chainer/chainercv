from chainercv.chainer_experimental.datasets.sliceable import GetterDataset


class TransformDataset(GetterDataset):
    """A sliceable version of :class:`chainer.datasets.TransformDataset`.

    Note that it reuqires :obj:`keys` to determine the names of returned
    values.

    Here is an example.

    >>> def transfrom(in_data):
    >>>     img, bbox, label = in_data
    >>>     ...
    >>>     return new_img, new_label
    >>>
    >>> dataset = TramsformDataset(dataset, ('img', 'label'), transform)
    >>> dataset.keys  # ('img', 'label')

    Args:
        dataset: The underlying dataset.
            This dataset should have :meth:`__len__` and :meth:`__getitem__`.
        keys (int or string or tuple of strings): The number or name(s) of
            data that the transform function returns.
        transform (callable): A function that is called to transform values
            returned by the underlying dataset's :meth:`__getitem__`.
    """

    def __init__(self, dataset, keys, transform):
        super(TransformDataset, self).__init__()
        self._dataset = dataset
        self._transform = transform
        if isinstance(keys, int):
            if keys == 1:
                keys = None
            else:
                keys = (None,) * keys
        self.add_getter(keys, self._get)

    def __len__(self):
        return len(self._dataset)

    def _get(self, index):
        return self._transform(self._dataset[index])
