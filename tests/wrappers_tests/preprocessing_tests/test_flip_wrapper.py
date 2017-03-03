import unittest

import numpy as np

from chainer import testing

import chainercv
from chainercv.testing import helper


class TestFlipWrapper(unittest.TestCase):

    def test_flip_wrapper(self):
        constant = np.arange(np.prod((3, 10, 10))).reshape(3, 10, 10)
        dataset = helper.DummyDataset(
            shapes=[(3, 10, 10), (3, 10, 10)], constants=[constant, constant])
        dataset = chainercv.wrappers.FlipWrapper(
            dataset, [0], orientation='h')
        img0, img1 = dataset[0]
        np.testing.assert_equal(img1, constant)


testing.run_module(__name__, __file__)
