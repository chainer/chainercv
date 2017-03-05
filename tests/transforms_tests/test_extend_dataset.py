import unittest

import numpy as np

from chainer import testing
from chainercv.testing import SimpleDataset
from chainercv.transforms import extend


class TestExtend(unittest.TestCase):

    def test_extend(self):
        dataset = SimpleDataset(np.random.uniform(size=(10, 3, 32, 32)))
        first_img = dataset.get_example(0)

        def transform(in_data):
            return in_data * 3

        extend(dataset, transform)
        out = dataset.get_example(0)
        np.testing.assert_equal(out, first_img * 3)


testing.run_module(__name__, __file__)
