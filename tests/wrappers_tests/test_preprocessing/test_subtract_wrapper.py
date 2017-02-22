import unittest

import numpy as np

from chainer import testing

import chainer_cv
from chainer_cv.testing import helper


class TestSubtractWrapper(unittest.TestCase):

    def test_subtract_wrapper(self):
        constant = np.ones((3, 10, 10))
        dataset = chainer_cv.wrappers.SubtractWrapper(
            helper.DummyDataset(
                shapes=[(3, 10, 10), (3, 10, 10)],
                constants=[constant, constant])
        )

        img0, img1 = dataset.get_example(0)
        img0_inside, img1_inside = dataset._dataset.get_example(0)

        img0_add = dataset.value + img0

        np.testing.assert_almost_equal(img0_inside, img0_add)
        np.testing.assert_almost_equal(img1_inside, img1)


testing.run_module(__name__, __file__)
