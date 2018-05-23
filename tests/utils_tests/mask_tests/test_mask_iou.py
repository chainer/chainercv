from __future__ import division

import unittest

import numpy as np

from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.utils import mask_iou


@testing.parameterize(
    {'mask_a': np.array(
        [[[False, False], [True, True]],
         [[True, True], [False, False]]],
        dtype=np.bool),
     'mask_b': np.array(
        [[[False, False], [True, True]],
         [[True, True], [False, False]],
         [[True, False], [True, True]],
         [[True, True], [False, True]]],
        dtype=np.bool),
     'expected': np.array(
        [[1., 0., 2 / 3, 1 / 4],
         [0., 1., 1 / 4, 2 / 3]],
        dtype=np.float32)
     },
    {'mask_a': np.array(
        [[[False, False], [True, True]],
         [[True, True], [False, False]],
         [[True, True], [True, False]],
         [[False, True], [True, True]]],
        dtype=np.bool),
     'mask_b': np.array(
        [[[False, False], [True, True]],
         [[True, True], [False, False]]],
        dtype=np.bool),
     'expected': np.array(
        [[1., 0.], [0., 1.], [1 / 4, 2 / 3], [2 / 3, 1 / 4]],
        dtype=np.float32)
     },
    {'mask_a': np.zeros((0, 2, 2), dtype=np.bool),
     'mask_b': np.array([[[False, False], [False, False]]], dtype=np.bool),
     'expected': np.zeros((0, 1), dtype=np.float32)
     },
)
class TestMaskIou(unittest.TestCase):

    def check(self, mask_a, mask_b, expected):
        iou = mask_iou(mask_a, mask_b)

        self.assertIsInstance(iou, type(expected))
        np.testing.assert_equal(
            cuda.to_cpu(iou),
            cuda.to_cpu(expected))

    def test_mask_iou_cpu(self):
        self.check(self.mask_a, self.mask_b, self.expected)

    @attr.gpu
    def test_mask_iou_gpu(self):
        self.check(
            cuda.to_gpu(self.mask_a),
            cuda.to_gpu(self.mask_b),
            cuda.to_gpu(self.expected))


@testing.parameterize(
    {'mask_a': np.array([[[False], [True, True]]], dtype=np.bool),
     'mask_b': np.array([[[False, False], [True, True]]], dtype=np.bool)
     },
    {'mask_a': np.array([[[False, False, True], [True, True]]], dtype=np.bool),
     'mask_b': np.array([[[False, False], [True, True]]], dtype=np.bool)
     },
    {'mask_a': np.array([[[False, False], [True, True]]], dtype=np.bool),
     'mask_b': np.array([[[False], [True, True]]], dtype=np.bool)
     },
    {'mask_a': np.array([[[False, False], [True, True]]], dtype=np.bool),
     'mask_b': np.array([[[False, False, True], [True, True]]], dtype=np.bool)
     },
)
class TestMaskIouInvalidShape(unittest.TestCase):

    def test_mask_iou_invalid(self):
        with self.assertRaises(IndexError):
            mask_iou(self.mask_a, self.mask_b)


testing.run_module(__name__, __file__)
