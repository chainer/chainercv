import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.links.model.faster_rcnn import bbox2loc
from chainercv.links.model.faster_rcnn import loc2bbox


def generate_bbox(n, img_size, min_length, max_length):
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox


class TestLocBboxConversions(unittest.TestCase):

    def setUp(self):
        self.src_bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.dst_bbox = np.array([[0, 0, 49, 29]], dtype=np.float32)
        self.loc = np.array([[0, 0, 0, 0]], dtype=np.float32)

    def check_bbox2loc(self, src_bbox, dst_bbox, loc):
        pred_loc = bbox2loc(src_bbox, dst_bbox)

        self.assertIsInstance(pred_loc, type(loc))
        np.testing.assert_equal(cuda.to_cpu(pred_loc),
                                cuda.to_cpu(loc))

    def test_bbox2loc_cpu(self):
        self.check_bbox2loc(
            self.src_bbox, self.dst_bbox, self.loc)

    @attr.gpu
    def test_bbox2loc_gpu(self):
        self.check_bbox2loc(
            cuda.to_gpu(self.src_bbox),
            cuda.to_gpu(self.dst_bbox),
            cuda.to_gpu(self.loc))

    def check_loc2bbox(self, raw_bbox, bbox, expected):
        pred_raw_bbox = loc2bbox(raw_bbox, bbox)

        self.assertIsInstance(pred_raw_bbox, type(expected))
        np.testing.assert_equal(
            cuda.to_cpu(pred_raw_bbox), cuda.to_cpu(expected))

    def test_loc2bbox_cpu(self):
        self.check_loc2bbox(
            self.src_bbox,
            self.loc,
            self.dst_bbox)

    @attr.gpu
    def test_loc2bbox_gpu(self):
        self.check_loc2bbox(
            cuda.to_gpu(self.src_bbox),
            cuda.to_gpu(self.loc),
            cuda.to_gpu(self.dst_bbox))


class TestDeltaEncodeDecodeConsistency(unittest.TestCase):

    def setUp(self):
        self.src_bbox = generate_bbox(8, (32, 64), 4, 16)
        self.dst_bbox = self.src_bbox + 1

    def check_bbox_loc_conversions_consistency(
            self, src_bbox, dst_bbox):
        bbox = bbox2loc(src_bbox, dst_bbox)
        out_raw_bbox = loc2bbox(src_bbox, bbox)

        np.testing.assert_almost_equal(
            cuda.to_cpu(out_raw_bbox), cuda.to_cpu(dst_bbox), decimal=5)

    def test_bbox_loc_conversions_consistency_cpu(self):
        self.check_bbox_loc_conversions_consistency(
            self.src_bbox, self.dst_bbox)

    @attr.gpu
    def test_bbox_loc_conversions_consistency_gpu(self):
        self.check_bbox_loc_conversions_consistency(
            cuda.to_gpu(self.src_bbox),
            cuda.to_gpu(self.dst_bbox))


testing.run_module(__name__, __file__)
