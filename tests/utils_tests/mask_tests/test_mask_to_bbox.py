from __future__ import division

import unittest

import numpy as np

from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.utils import mask_to_bbox


@testing.parameterize(
    {'mask': np.array(
        [[[False, False, False, False],
          [False, True, True, True],
          [False, True, True, True]
          ]]),
     'expected': np.array([[1, 1, 3, 4]], dtype=np.float32)
     },
    {'mask': np.array(
        [[[False, False],
          [False, True]],
         [[True, False],
          [False, True]]]),
     'expected': np.array([[1, 1, 2, 2], [0, 0, 2, 2]], dtype=np.float32)
     },
    {'mask': np.array(
        [[[False, False],
          [False, False]],
         [[True, False],
          [False, True]]]),
     'expected': np.array([[0, 0, 0, 0], [0, 0, 2, 2]], dtype=np.float32)
     },
)
class TestMaskToBbox(unittest.TestCase):

    def check(self, mask, expected):
        bbox = mask_to_bbox(mask)

        self.assertIsInstance(bbox, type(expected))
        self.assertEqual(bbox.dtype, expected.dtype)
        np.testing.assert_equal(
            cuda.to_cpu(bbox),
            cuda.to_cpu(expected))

    def test_mask_to_bbox_cpu(self):
        self.check(self.mask, self.expected)

    @attr.gpu
    def test_mask_to_bbox_gpu(self):
        self.check(
            cuda.to_gpu(self.mask),
            cuda.to_gpu(self.expected))


testing.run_module(__name__, __file__)
