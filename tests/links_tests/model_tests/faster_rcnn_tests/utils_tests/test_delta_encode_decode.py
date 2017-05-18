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
        self.raw_bbox_src = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.raw_bbox_dst = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.expected_enc = np.array([[0, 0, 0, 0]], dtype=np.float32)

    def check_delta_encode(self, raw_bbox_src, raw_bbox_dst, expected_enc):
        bbox = delta_encode(raw_bbox_src, raw_bbox_dst)

        self.assertIsInstance(bbox, type(expected_enc))
        np.testing.assert_equal(cuda.to_cpu(bbox),
                                cuda.to_cpu(expected_enc))

    def test_delta_encode_cpu(self):
        self.check_delta_encode(
            self.raw_bbox_src, self.raw_bbox_dst, self.expected_enc)

    @attr.gpu
    def test_delta_encode_gpu(self):
        self.check_delta_encode(
            cuda.to_gpu(self.raw_bbox_src),
            cuda.to_gpu(self.raw_bbox_dst),
            cuda.to_gpu(self.expected_enc))

    def check_delta_decode(self, raw_bbox, bbox, expected):
        pred_raw_bbox = delta_decode(raw_bbox, bbox)

        self.assertIsInstance(pred_raw_bbox, type(expected))
        np.testing.assert_equal(
            cuda.to_cpu(pred_raw_bbox), cuda.to_cpu(expected))

    def test_delta_decode_cpu(self):
        self.check_delta_decode(
            self.raw_bbox_src,
            self.expected_enc,
            self.raw_bbox_dst)

    @attr.gpu
    def test_delta_decode_gpu(self):
        self.check_delta_decode(
            cuda.to_gpu(self.raw_bbox_src),
            cuda.to_gpu(self.expected_enc),
            cuda.to_gpu(self.raw_bbox_dst))


class TestDeltaEncodeDecodeConsistency(unittest.TestCase):

    def setUp(self):
        self.raw_bbox_src = generate_bbox(8, (32, 64), 4, 16)
        self.raw_bbox_dst = self.raw_bbox_src + 1

    def check_delta_encode_decode_consistency(
            self, raw_bbox_src, raw_bbox_dst):
        bbox = delta_encode(raw_bbox_src, raw_bbox_dst)
        out_raw_bbox = delta_decode(raw_bbox_src, bbox)

        np.testing.assert_almost_equal(
            cuda.to_cpu(out_raw_bbox), cuda.to_cpu(raw_bbox_dst), decimal=5)

    def test_delta_encde_decode_consistency_cpu(self):
        self.check_delta_encode_decode_consistency(
            self.raw_bbox_src, self.raw_bbox_dst)

    @attr.gpu
    def test_delta_encode_decode_consistency_gpu(self):
        self.check_delta_encode_decode_consistency(
            cuda.to_gpu(self.raw_bbox_src),
            cuda.to_gpu(self.raw_bbox_dst))


testing.run_module(__name__, __file__)
