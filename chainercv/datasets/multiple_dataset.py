class MultipleDataset(object):

    """Dataset that indexes data of some base datasets.

    This dataset wraps some base datasets.

    .. note::

        The index used to access data is either an integer or a slice. If it
        is a slice, the base dataset is assumed to return a list of outputs
        each corresponding to the output of the integer indexing.

    Args:
        datasets: Underlying datasets.

    """

    def __init__(self, datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(map(len, self._datasets))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[i] for i in range(index.start, index.stop, index.step)]
        else:
            if index < 0:
                raise IndexError
            for dataset in self._datasets:
                if index < len(dataset):
                    return dataset[index]
                index -= len(dataset)
            raise IndexError
