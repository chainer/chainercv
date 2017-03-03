import unittest

import numpy as np

from chainer import testing

import chainercv
from chainercv.testing import helper


class TestKeepSubsetWrapper(unittest.TestCase):

    def test_keep_subset_wrapper(self):
        array = np.random.uniform(size=(100, 3, 10, 10)).astype(np.float32)
        dataset = helper.SimpleDataset(array)

        indices = range(10, 0, -1)
        wrapped_dataset = chainercv.wrappers.KeepSubsetWrapper(
            dataset, indices)

        for i, idx in enumerate(indices):
            src = wrapped_dataset[i]
            dst = array[idx]
            np.testing.assert_equal(src, dst)


testing.run_module(__name__, __file__)
