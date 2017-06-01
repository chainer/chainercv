import unittest

import numpy as np

from chainer import cuda

from chainer import testing
from chainer.testing import attr

from chainercv.utils import generate_random_bbox
from chainercv.utils import non_maximum_suppression


@testing.parameterize(
    {'threshold': 1, 'expect': np.array([0, 1, 2, 3])},
    {'threshold': 0.5, 'expect': np.array([0, 1, 3])},
    {'threshold': 0.3, 'expect': np.array([0, 2, 3])},
    {'threshold': 0.2, 'expect': np.array([0, 3])},
    {'threshold': 0, 'expect': np.array([0])},
)
class TestNonMaximumSuppression(unittest.TestCase):

    def setUp(self):
        self.bbox = np.array((
            (0, 0, 4, 4),
            (1, 1, 5, 5),  # 9/23
            (2, 1, 6, 5),  # 6/26, 12/20
            (4, 0, 8, 4),  # 0/32, 3/29, 6/26
        ))

    def check_non_maximum_suppression(self, bbox, threshold, expect):
        selec = non_maximum_suppression(bbox, threshold)
        self.assertIsInstance(selec, type(bbox))
        self.assertEqual(selec.dtype, np.int32)
        np.testing.assert_equal(
            cuda.to_cpu(selec),
            cuda.to_cpu(expect))

    def test_non_maximum_suppression_cpu(self):
        self.check_non_maximum_suppression(
            self.bbox, self.threshold, self.expect)

    @attr.gpu
    def test_non_maximum_suppression_gpu(self):
        self.check_non_maximum_suppression(
            cuda.to_gpu(self.bbox),
            self.threshold,
            cuda.to_gpu(self.expect)
        )


class TestNonMaximumSuppressionConsistency(unittest.TestCase):

    @attr.gpu
    def test_non_maximum_suppression_consistency(self):
        bbox = generate_random_bbox(6000, (600, 800), 32, 512)

        cpu_selec = non_maximum_suppression(bbox, 0.5)
        gpu_selec = non_maximum_suppression(cuda.to_gpu(bbox), 0.5)

        np.testing.assert_equal(cpu_selec, cuda.to_cpu(gpu_selec))


class TestNonMaximumSuppressionOptions(unittest.TestCase):

    def setUp(self):
        self.bbox = generate_random_bbox(6000, (600, 800), 32, 512)
        self.score = np.random.uniform(0, 100, size=(len(self.bbox),))
        self.limit = 100
        self.threshold = 0.5

    def check_non_maximum_suppression_options(
            self, bbox, threshold, score, limit):
        # Pass all options to the tested function
        scored_selec = non_maximum_suppression(bbox, threshold, score, limit)
        self.assertIsInstance(scored_selec, type(bbox))

        # Reorder inputs befor passing it to the function.
        # Reorder the outputs according to scores.
        # CuPy does not support argsort
        order = cuda.to_cpu(score).argsort()[::-1]
        reordered_selec = non_maximum_suppression(
            bbox[order], threshold, score=None, limit=None)
        reordered_selec = reordered_selec[:limit]
        reordered_selec = cuda.to_cpu(reordered_selec)
        reordered_selec = order[reordered_selec]

        np.testing.assert_equal(
            cuda.to_cpu(scored_selec), cuda.to_cpu(reordered_selec))

    def test_non_maximum_suppression_options_cpu(self):
        self.check_non_maximum_suppression_options(
            self.bbox, self.threshold, self.score, self.limit)

    @attr.gpu
    def test_non_maximum_suppression_options_gpu(self):
        self.check_non_maximum_suppression_options(
            cuda.to_gpu(self.bbox),
            self.threshold, cuda.to_gpu(self.score), self.limit)


class TestNonMaximumSuppressionZeroLengthBbox(unittest.TestCase):

    def setUp(self):
        self.bbox = np.zeros((0, 4))

    def check_non_maximum_suppression_zero_legnth_bbox(
            self, bbox, threshold):
        selec = non_maximum_suppression(bbox, threshold)
        self.assertIsInstance(selec, type(bbox))
        self.assertEqual(selec.shape, (0,))

    def test_non_maximum_suppression_zero_length_bbox_cpu(self):
        self.check_non_maximum_suppression_zero_legnth_bbox(
            self.bbox, 0.5)

    @attr.gpu
    def test_non_maximum_suppression_zero_length_bbox_gpu(self):
        self.check_non_maximum_suppression_zero_legnth_bbox(
            cuda.to_gpu(self.bbox), 0.5)


testing.run_module(__name__, __file__)
