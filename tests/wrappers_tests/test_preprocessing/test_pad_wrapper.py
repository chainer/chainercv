import unittest

import numpy as np

import chainer_cv
from chainer_cv.testing import helper


class TestPadWrapper(unittest.TestCase):

    def setUp(self):
        constant = np.ones((2, 3, 3))
        self.dataset = helper.DummyDataset(
            shapes=[(2, 3, 3), (2, 3, 3)], constants=[constant, constant])

    def test_pad_wrapper1(self):
        wrapped = chainer_cv.wrappers.PadWrapper(
            self.dataset, (5, 5), [0], bg_values=-1)
        img0, img1 = wrapped.get_example(0)
        self.assertEqual(img0.shape, (2, 5, 5))
        self.assertEqual(img1.shape, (2, 3, 3))
        np.testing.assert_equal(img0[0, 0], -1)

    def test_pad_wrapper2(self):
        wrapped = chainer_cv.wrappers.PadWrapper(
            self.dataset, (5, 5), [0, 1], bg_values={0: -1, 1: -2})
        img0, img1 = wrapped.get_example(0)
        self.assertEqual(img0.shape, (2, 5, 5))
        self.assertEqual(img1.shape, (2, 5, 5))
        np.testing.assert_equal(img0[0, 0], -1)
        np.testing.assert_equal(img1[0, 0], -2)
