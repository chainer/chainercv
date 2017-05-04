import unittest

import numpy as np

from chainer import testing

from chainercv.links.utils import bbox_overlap


class TestBboxOverlap(unittest.TestCase):

    def test_bbox_overlap(self):
        bbox = np.array([[0, 0, 8, 8]], dtype=np.float32)
        query_bbox = np.array([[3, 5, 10, 12], [9, 10, 11, 12]],
                              dtype=np.float32)

        overlap = bbox_overlap(bbox, query_bbox)

        o0 = float(4 * 6) / (9 * 9 + 8 * 8 - 4 * 6)
        o1 = 0.
        expected = np.array([[o0, o1]], dtype=query_bbox.dtype)
        np.testing.assert_equal(overlap, expected)


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
