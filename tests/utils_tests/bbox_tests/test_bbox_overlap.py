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
        overlap = bbox_overlap(bbox_a, bbox_b)

        self.assertIsInstance(overlap, type(expected))
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


@testing.parameterize(
    {'bbox_a': [[0, 0, 8]], 'bbox_b': [[1, 1, 9, 9]]},
    {'bbox_a': [[0, 0, 8, 0, 1]], 'bbox_b': [[1, 1, 9, 9]]},
    {'bbox_a': [[0, 0, 8, 8]], 'bbox_b': [[1, 1, 9]]},
    {'bbox_a': [[0, 0, 8, 8]], 'bbox_b': [[1, 1, 9, 9, 10]]}
)
class TestBboxOverlapInvalidShape(unittest.TestCase):

    def test_bbox_overlap_invalid(self):
        bbox_a = np.array(self.bbox_a, dtype=np.float32)
        bbox_b = np.array(self.bbox_b, dtype=np.float32)

        with self.assertRaises(IndexError):
            bbox_overlap(bbox_a, bbox_b)


testing.run_module(__name__, __file__)
