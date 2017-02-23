import numpy as np

import chainer


class DummyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, shapes, length=100, constants=None, dtypes=None):
        self.length = length

        if not all(isinstance(elem, tuple) for elem in shapes):
            raise ValueError('`shapes` needs to be a list of tuples')
        self.shapes = shapes
        self.n_arrays = len(self.shapes)

        if constants is None:
            constants = [None] * self.n_arrays
        self.constants = constants

        if dtypes is None:
            dtypes = self.n_arrays * [np.float64]
        if len(dtypes) != self.n_arrays:
            raise ValueError('dtypes is either None or iterable whose length '
                             'is equal to shapes')
        self.dtypes = dtypes

    def __len__(self):
        return self.length

    def get_example(self, i):
        return self._get_example(i, self.shapes, self.dtypes, self.constants)

    def _get_example(self, i, shapes, dtypes, constants):
        out = []
        for shape, dtype, constant in zip(shapes, dtypes, constants):
            if constant is None:
                a = np.random.uniform(size=(shape))
            else:
                a = constant.copy()
                a = np.broadcast_to(a, shape)
            a = a.astype(dtype)
            out.append(a)
        return tuple(out)


class DummyDatasetGetRawData(DummyDataset):

    def __init__(self, shapes, get_raw_data_shapes, length=100,
                 constants=None, dtypes=None, get_raw_data_dtypes=None):
        super(DummyDatasetGetRawData, self).__init__(
            shapes, length, constants, dtypes)

        if len(get_raw_data_shapes) != self.n_arrays:
            raise ValueError('get_raw_data_shapes is either None or iterable '
                             'whose length is equal to shapes')
        self.get_raw_data_shapes = get_raw_data_shapes
        self.n_get_raw_data = len(get_raw_data_shapes)

        if len(get_raw_data_dtypes) != self.n_get_raw_data:
            raise ValueError('dtypes is either None or iterable whose length '
                             'is equal to shapes')
        self.get_raw_data_dtypes = get_raw_data_dtypes

    def get_raw_data(self, i):
        return self._get_example(i, self.get_raw_data_shapes,
                                 self.get_raw_data_dtypes, self.constants)


class SimpleDataset(chainer.dataset.DatasetMixin):

    def __init__(self, input_array):
        self.input_array = input_array

    def __len__(self):
        return len(self.input_array)

    def get_example(self, i):
        return self.input_array[i]


class ConstantReturnModel(chainer.Chain):
    """A chainer.Chain that returns constant values

    This is a ``chainer.Chain`` instance that returns `return_value` when
    a method `__call__` is called.

    """

    def __init__(self, return_value):
        super(ConstantReturnModel, self).__init__()
        self.return_value = return_value

    def __call__(self, *args):
        return self.return_value
