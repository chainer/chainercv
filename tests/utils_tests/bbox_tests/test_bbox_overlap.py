from __future__ import division

import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.utils import bbox_overlap


class TestBboxOverlap(unittest.TestCase):

    def setUp(self):
        self.bbox = np.array([[0, 0, 8, 8]], dtype=np.float32)
        self.query_bbox = np.array([[3, 5, 10, 12], [9, 10, 11, 12]],
                                   dtype=np.float32)

        o0 = (5 * 3) / (8 * 8 + 7 * 7 - 5 * 3)
        o1 = 0.
        self.expected = np.array([[o0, o1]], dtype=np.float32)

    def check(self, bbox, query_bbox, expected):
        xp = cuda.get_array_module(bbox)
        overlap = bbox_overlap(bbox, query_bbox)

        self.assertIsInstance(overlap, xp.ndarray)
        np.testing.assert_equal(
            cuda.to_cpu(overlap),
            cuda.to_cpu(expected))

    def test_bbox_overlap_cpu(self):
        self.check(self.bbox, self.query_bbox, self.expected)

    @attr.gpu
    def test_bbox_overlap_gpu(self):
        self.check(
            cuda.to_gpu(self.bbox),
            cuda.to_gpu(self.query_bbox),
            cuda.to_gpu(self.expected))


class TestBboxOverlapInvalidShape(unittest.TestCase):

    def test_bbox_overlap_invalid_bbox(self):
        bbox = np.array([[0, 0, 8]], dtype=np.float32)
        query_bbox = np.array([[1, 1, 9, 9]], dtype=np.float32)

        with self.assertRaises(IndexError):
            bbox_overlap(bbox, query_bbox)

    def test_bbox_overlap_invalid_query_bbox(self):
        bbox = np.array([[0, 0, 8, 8]], dtype=np.float32)
        query_bbox = np.array([[1, 1, 9]], dtype=np.float32)

        with self.assertRaises(IndexError):
            bbox_overlap(bbox, query_bbox)


testing.run_module(__name__, __file__)
