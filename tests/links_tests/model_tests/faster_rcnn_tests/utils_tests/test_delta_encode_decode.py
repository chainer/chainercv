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
        self.raw_bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.base_raw_bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.expected_enc = np.array([[0, 0, 0, 0]], dtype=np.float32)

    def check_delta_encode(self, raw_bbox, base_raw_bbox, expected_enc):
        bbox = delta_encode(raw_bbox, base_raw_bbox)

        self.assertIsInstance(bbox, type(expected_enc))
        np.testing.assert_equal(cuda.to_cpu(bbox),
                                cuda.to_cpu(expected_enc))

    def test_delta_encode_cpu(self):
        self.check_delta_encode(
            self.raw_bbox, self.base_raw_bbox, self.expected_enc)

    @attr.gpu
    def test_delta_encode_gpu(self):
        self.check_delta_encode(
            cuda.to_gpu(self.raw_bbox),
            cuda.to_gpu(self.base_raw_bbox),
            cuda.to_gpu(self.expected_enc))

    def check_delta_decode(self, bbox, base_raw_bbox, expected):
        raw_bbox = delta_decode(bbox, base_raw_bbox)

        self.assertIsInstance(raw_bbox, type(expected))
        np.testing.assert_equal(
            cuda.to_cpu(raw_bbox), cuda.to_cpu(expected))

    def test_delta_decode_cpu(self):
        self.check_delta_decode(
            self.expected_enc,
            self.raw_bbox,
            self.base_raw_bbox)

    @attr.gpu
    def test_delta_decode_gpu(self):
        self.check_delta_decode(
            cuda.to_gpu(self.expected_enc),
            cuda.to_gpu(self.raw_bbox),
            cuda.to_gpu(self.base_raw_bbox))


class TestDeltaEncodeDecodeConsistency(unittest.TestCase):

    def setUp(self):
        self.raw_bbox = generate_bbox(8, (32, 64), 4, 16)
        self.base_raw_bbox = self.raw_bbox + 1

    def check_delta_encode_decode_consistency(self, raw_bbox, base_raw_bbox):
        bbox = delta_encode(raw_bbox, base_raw_bbox)
        out_raw_bbox = delta_decode(bbox, raw_bbox)

        np.testing.assert_almost_equal(
            cuda.to_cpu(out_raw_bbox), cuda.to_cpu(base_raw_bbox), decimal=5)

    def test_delta_encde_decode_consistency_cpu(self):
        self.check_delta_encode_decode_consistency(
            self.raw_bbox, self.base_raw_bbox)

    @attr.gpu
    def test_delta_encode_decode_consistency_gpu(self):
        self.check_delta_encode_decode_consistency(
            cuda.to_gpu(self.raw_bbox),
            cuda.to_gpu(self.base_raw_bbox))


testing.run_module(__name__, __file__)
