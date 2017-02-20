import unittest

import numpy as np

from chainer import testing

import chainer_cv
from chainer_cv.testing import helper


class TestRandomMirrorWrapper(unittest.TestCase):

    def test_random_mirror_wrapper(self):
        constant = np.arange(np.prod((3, 10, 10))).reshape(3, 10, 10)
        dataset = helper.DummyDataset(
            shapes=[(3, 10, 10), (3, 10, 10)], constant=constant)
        dataset = chainer_cv.wrappers.RandomMirrorWrapper(
            dataset, [0], orientation='h')
        img0, img1 = dataset.get_example(0)
        np.testing.assert_equal(img1, constant)


testing.run_module(__name__, __file__)
