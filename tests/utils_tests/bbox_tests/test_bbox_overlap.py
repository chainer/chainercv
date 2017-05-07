from __future__ import division

import unittest

import numpy as np

from chainer import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.utils import bbox_overlap


@testing.parameterize(
    {'bbox_a': np.array([[0, 0, 8, 8]], dtype=np.float32),
     'bbox_b': np.array(
         [[3, 5, 10, 12], [9, 10, 11, 12], [0, 0, 8, 8]], dtype=np.float32),
     'expected': np.array(
         [[(5 * 3) / (8 * 8 + 7 * 7 - 5 * 3), 0., 1.]], dtype=np.float32)
     },
    {'bbox_a': np.array(
         [[3, 5, 10, 12], [9, 10, 11, 12], [0, 0, 8, 8]], dtype=np.float32),
     'bbox_b': np.array([[0, 0, 8, 8]], dtype=np.float32),
     'expected': np.array(
         [[(5 * 3) / (8 * 8 + 7 * 7 - 5 * 3)], [0.], [1.]], dtype=np.float32)
     },
    {'bbox_a': np.zeros((0, 4), dtype=np.float32),
     'bbox_b': np.array([[0, 0, 1, 1]], dtype=np.float32),
     'expected': np.zeros((0, 1), dtype=np.float32)
     },
)
class TestBboxOverlap(unittest.TestCase):

    def check(self, bbox_a, bbox_b, expected):
        xp = cuda.get_array_module(bbox_a)
        overlap = bbox_overlap(bbox_a, bbox_b)

        self.assertIsInstance(overlap, xp.ndarray)
        np.testing.assert_equal(
            cuda.to_cpu(overlap),
            cuda.to_cpu(expected))

    def test_bbox_overlap_cpu(self):
        self.check(self.bbox_a, self.bbox_b, self.expected)

    @attr.gpu
    def test_bbox_overlap_gpu(self):
        self.check(
            cuda.to_gpu(self.bbox_a),
            cuda.to_gpu(self.bbox_b),
            cuda.to_gpu(self.expected))


class TestBboxOverlapInvalidShape(unittest.TestCase):

    def test_bbox_overlap_invalid_bbox(self):
        bbox_a = np.array([[0, 0, 8]], dtype=np.float32)
        bbox_b = np.array([[1, 1, 9, 9]], dtype=np.float32)

        with self.assertRaises(IndexError):
            bbox_overlap(bbox_a, bbox_b)

    def test_bbox_overlap_invalid_query_bbox(self):
        bbox_a = np.array([[0, 0, 8, 8]], dtype=np.float32)
        bbox_b = np.array([[1, 1, 9]], dtype=np.float32)

        with self.assertRaises(IndexError):
            bbox_overlap(bbox_a, bbox_b)


testing.run_module(__name__, __file__)
