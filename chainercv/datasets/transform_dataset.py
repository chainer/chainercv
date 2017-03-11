class TransformDataset(object):

    """Dataset that indexes data of a base dataset and transforms it.

    This dataset wraps a base dataset by modifying the behavior of the base
    dataset's :meth:`__getitem__`. Arrays returned by :meth:`__getitem__` of
    the base dataset with an integer index are transformed by the given
    function :obj:`transform`.

    The function :obj:`transform` takes, as an argument, :obj:`in_data`, which
    is output of the base dataset's :meth:`__getitem__`, and returns
    transformed arrays as output. Please see the following example.

    >>> from chainer.datasets import get_mnist
    >>> from chainercv.datasets import TransformDataset
    >>> dataset, _ = get_mnist()
    >>> def transform(in_data):
    >>>     img, label = in_data
    >>>     img -= 0.5  # scale to [-0.5, -0.5]
    >>>     return img, label
    >>> dataset = TransformDataset(dataset, transform)

    .. note::

        The index used to access data is either an integer or a slice. If it
        is a slice, the base dataset is assumed to return a list of outputs
        each corresponding to the output of the integer indexing.

    Args:
        dataset: Underlying dataset. The index of this dataset corresponds
            to the index of the base dataset.
        transform (callable): A function that is called to transform values
            returned by the underlying dataset's :meth:`__getitem__`.

    """

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        in_data = self._dataset[index]
        if isinstance(index, slice):
            return [self._transform(in_data_elem) for in_data_elem in in_data]
        else:
            return self._transform(in_data)

    def __len__(self):
        return len(self._dataset)
