import unittest

import numpy as np

from chainer import testing
from chainercv.datasets import TransformDataset


class TestTransformDataset(unittest.TestCase):

    def test_transform_dataset(self):
        dataset = np.random.uniform(size=(10, 3, 32, 32))
        first_img = dataset[0]

        def transform(in_data):
            return in_data * 3

        transformed_dataset = TransformDataset(dataset, transform)
        out = transformed_dataset[0]
        np.testing.assert_equal(out, first_img * 3)

        outs = transformed_dataset[:2]
        np.testing.assert_equal(
            [transform(elem) for elem in dataset[:2]], outs)


testing.run_module(__name__, __file__)
