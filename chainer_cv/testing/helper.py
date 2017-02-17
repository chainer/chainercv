import chainer
import numpy as np


class DummyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, shape=(3, 10, 10), n_arrays=2, length=100,
                 constant=None):
        self.shape = shape
        self.n_arrays = n_arrays
        self.length = length
        self.constant = constant

    def __len__(self):
        return self.length

    def get_example(self, i):
        out = []
        for i in range(self.n_arrays):
            if self.constant is None:
                a = np.random.uniform(size=(self.shape))
            else:
                a = np.ones(shape=self.shape)
                a *= self.constant
            out.append(a)
        return tuple(out)
