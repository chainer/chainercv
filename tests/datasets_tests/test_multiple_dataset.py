import unittest

import numpy as np

from chainer import testing
from chainercv.datasets import MultipleDataset


class TestMultipleDataset(unittest.TestCase):

    def test_transform_dataset(self):
        dataset0 = np.random.uniform(size=(5, 3, 32, 32))
        dataset1 = np.random.uniform(size=(15, 3, 32, 32))
        multiple_dataset = MultipleDataset([dataset0, dataset1])

        self.assertEqual(len(multiple_dataset), 20)

        np.testing.assert_equal(multiple_dataset[0], dataset0[0])
        np.testing.assert_equal(multiple_dataset[4], dataset0[4])
        np.testing.assert_equal(multiple_dataset[5], dataset1[0])
        np.testing.assert_equal(multiple_dataset[8], dataset1[3])

    def test_transform_dataset_slice(self):
        dataset0 = np.random.uniform(size=(5, 3, 32, 32))
        dataset1 = np.random.uniform(size=(15, 3, 32, 32))
        multiple_dataset = MultipleDataset([dataset0, dataset1])

        self.assertEqual(len(multiple_dataset), 20)

        out = multiple_dataset[1:8:2]
        self.assertEqual(len(out), 4)
        np.testing.assert_equal(out[0], dataset0[1])
        np.testing.assert_equal(out[1], dataset0[3])
        np.testing.assert_equal(out[2], dataset1[0])
        np.testing.assert_equal(out[3], dataset1[2])


testing.run_module(__name__, __file__)
