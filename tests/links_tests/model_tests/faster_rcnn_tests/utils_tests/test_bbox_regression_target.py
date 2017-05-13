import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links import bbox_regression_target
from chainercv.links import bbox_regression_target_inv


def generate_bbox(n, img_size, min_length, max_length):
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


class TestBboxRegressionTarget(unittest.TestCase):

    def setUp(self):
        self.bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.gt_bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.target = np.array([[0, 0, 0, 0]], dtype=np.float32)

    def check_bbox_regression_target(self, bbox, gt_bbox, target):
        out_target = bbox_regression_target(bbox, gt_bbox)

        xp = cuda.get_array_module(target)
        self.assertEqual(xp, cuda.get_array_module(out_target))

        np.testing.assert_equal(cuda.to_cpu(out_target),
                                cuda.to_cpu(target))

    def test_bbox_regression_target_cpu(self):
        self.check_bbox_regression_target(self.bbox, self.gt_bbox, self.target)

    @attr.gpu
    def test_bbox_regression_target_gpu(self):
        self.check_bbox_regression_target(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.gt_bbox),
            cuda.to_gpu(self.target))

    def check_bbox_regression_target_inv(self, bbox, gt_bbox, target):
        out_bbox = bbox_regression_target_inv(bbox, target)

        xp = cuda.get_array_module(gt_bbox)
        self.assertEqual(xp, cuda.get_array_module(out_bbox))

        np.testing.assert_equal(
            cuda.to_cpu(out_bbox), cuda.to_cpu(gt_bbox))

    def test_bbox_regression_target_inv_cpu(self):
        self.check_bbox_regression_target_inv(
            self.bbox,
            self.gt_bbox,
            self.target)

    @attr.gpu
    def test_bbox_regression_target_inv_gpu(self):
        self.check_bbox_regression_target_inv(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.gt_bbox),
            cuda.to_gpu(self.target))


class TestBboxRegressionTargetConsistency(unittest.TestCase):

    def setUp(self):
        self.bbox = generate_bbox(8, (32, 64), 4, 16)
        self.gt_bbox = self.bbox + 1

    def check_bbox_regression_target_consistency(self, bbox, gt_bbox):
        target = bbox_regression_target(bbox, gt_bbox)
        out_bbox = bbox_regression_target_inv(bbox, target)

        np.testing.assert_almost_equal(
            cuda.to_cpu(out_bbox), cuda.to_cpu(gt_bbox), decimal=5)

    def test_bbox_regression_target_consistency_cpu(self):
        self.check_bbox_regression_target_consistency(
            self.bbox, self.gt_bbox)

    @attr.gpu
    def test_bbox_regression_target_consistency_gpu(self):
        self.check_bbox_regression_target_consistency(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.gt_bbox))


testing.run_module(__name__, __file__)
