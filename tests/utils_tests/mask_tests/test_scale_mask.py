from __future__ import division

import unittest

import numpy as np
import PIL.Image

from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr

from chainercv.transforms import resize
from chainercv.utils import generate_random_bbox
from chainercv.utils import mask_to_bbox
from chainercv.utils import scale_mask


@testing.parameterize(
    {'mask': np.array(
        [[[False, False],
          [False, True]]]),
     'expected': np.array(
         [[[False, False, False, False],
           [False, False, False, False],
           [False, False, True, True],
           [False, False, True, True]]])
     }
)
class TestScaleMaskSimple(unittest.TestCase):

    def check(self, mask, expected):
        in_type = type(mask)
        bbox = mask_to_bbox(mask)
        size = 4
        out_mask = scale_mask(mask, bbox, size)

        self.assertIsInstance(out_mask, in_type)
        self.assertEqual(out_mask.dtype, np.bool)

        np.testing.assert_equal(
            cuda.to_cpu(out_mask),
            cuda.to_cpu(expected))

    def test_scale_mask_simple_cpu(self):
        self.check(self.mask, self.expected)

    @attr.gpu
    def test_scale_mask_simple_gpu(self):
        self.check(cuda.to_gpu(self.mask), cuda.to_gpu(self.expected))


class TestScaleMaskCompareResize(unittest.TestCase):

    def test(self):
        H = 80
        W = 90
        n_inst = 10

        mask = np.zeros((n_inst, H, W), dtype=np.bool)
        bbox = generate_random_bbox(n_inst, (H, W), 10, 30).astype(np.int32)
        for i, bb in enumerate(bbox):
            y_min, x_min, y_max, x_max = bb
            m = np.random.randint(0, 2, size=(y_max - y_min, x_max - x_min))
            m[5, 5] = 1  # At least one element is one
            mask[i, y_min:y_max, x_min:x_max] = m
        bbox = mask_to_bbox(mask)
        size = H * 2
        out_H = size
        out_W = W * 2
        out_mask = scale_mask(mask, bbox, size)

        expected = resize(
            mask.astype(np.float32), (out_H, out_W),
            interpolation=PIL.Image.NEAREST).astype(np.bool)
        np.testing.assert_equal(out_mask, expected)


testing.run_module(__name__, __file__)
