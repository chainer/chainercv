import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links import delta_decode
from chainercv.links import delta_encode


def generate_bbox(n, img_size, min_length, max_length):
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


class TestDeltaEncodeDecode(unittest.TestCase):

    def setUp(self):
        self.bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.gt_bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.target = np.array([[0, 0, 0, 0]], dtype=np.float32)

    def check_delta_encode(self, bbox, gt_bbox, target):
        out_target = delta_encode(bbox, gt_bbox)

        xp = cuda.get_array_module(target)
        self.assertEqual(xp, cuda.get_array_module(out_target))

        np.testing.assert_equal(cuda.to_cpu(out_target),
                                cuda.to_cpu(target))

    def test_delta_encode_cpu(self):
        self.check_delta_encode(self.bbox, self.gt_bbox, self.target)

    @attr.gpu
    def test_delta_encode_gpu(self):
        self.check_delta_encode(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.gt_bbox),
            cuda.to_gpu(self.target))

    def check_delta_decode(self, bbox, gt_bbox, target):
        out_bbox = delta_decode(bbox, target)

        xp = cuda.get_array_module(gt_bbox)
        self.assertEqual(xp, cuda.get_array_module(out_bbox))

        np.testing.assert_equal(
            cuda.to_cpu(out_bbox), cuda.to_cpu(gt_bbox))

    def test_delta_decode_cpu(self):
        self.check_delta_decode(
            self.bbox,
            self.gt_bbox,
            self.target)

    @attr.gpu
    def test_delta_decode_gpu(self):
        self.check_delta_decode(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.gt_bbox),
            cuda.to_gpu(self.target))


class TestDeltaEncodeDecodeConsistency(unittest.TestCase):

    def setUp(self):
        self.bbox = generate_bbox(8, (32, 64), 4, 16)
        self.gt_bbox = self.bbox + 1

    def check_delta_encode_decode_consistency(self, bbox, gt_bbox):
        target = delta_encode(bbox, gt_bbox)
        out_bbox = delta_decode(bbox, target)

        np.testing.assert_almost_equal(
            cuda.to_cpu(out_bbox), cuda.to_cpu(gt_bbox), decimal=5)

    def test_delta_encde_decode_consistency_cpu(self):
        self.check_delta_encode_decode_consistency(
            self.bbox, self.gt_bbox)

    @attr.gpu
    def test_delta_encode_decode_consistency_gpu(self):
        self.check_delta_encode_decode_consistency(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.gt_bbox))


testing.run_module(__name__, __file__)
