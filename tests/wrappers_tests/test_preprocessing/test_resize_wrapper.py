import unittest

import numpy as np

import chainer_cv
from chainer_cv.testing import helper


class TestResizeWrapper(unittest.TestCase):

    def test_resize_wrapper1(self):
        # soft_min
        dataset = helper.SimpleDataset(
            np.random.uniform(size=(100, 3, 32, 64)))
        wrapped = chainer_cv.wrappers.ResizeWrapper(
            dataset, [0],
            output_shape=chainer_cv.wrappers.output_shape_soft_min_hard_max(
                48, 120))

        img0 = wrapped.get_example(0)
        self.assertEqual(img0.shape, (3, 48, 96))

    def test_resize_wrapper2(self):
        # hard_max
        dataset = helper.SimpleDataset(
            np.random.uniform(size=(100, 3, 32, 64)))
        wrapped = chainer_cv.wrappers.ResizeWrapper(
            dataset, [0],
            output_shape=chainer_cv.wrappers.output_shape_soft_min_hard_max(
                32, 32))

        img0 = wrapped.get_example(0)
        self.assertEqual(img0.shape, (3, 16, 32))

    def test_resize_wrapper3(self):
        # none of hard_max and soft_min
        dataset = helper.SimpleDataset(
            np.random.uniform(size=(100, 3, 32, 64)))
        wrapped = chainer_cv.wrappers.ResizeWrapper(
            dataset, [0],
            output_shape=chainer_cv.wrappers.output_shape_soft_min_hard_max(
                32, 96))

        img0 = wrapped.get_example(0)
        self.assertEqual(img0.shape, (3, 32, 64))
