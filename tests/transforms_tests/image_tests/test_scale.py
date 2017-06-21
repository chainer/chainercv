import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import scale


@testing.parameterize(
    {'in_shape': (3, 24, 16), 'size': 8,
     'fit_short': True, 'out_shape': (3, 12, 8)},
    {'in_shape': (3, 16, 24), 'size': 8,
     'fit_short': True, 'out_shape': (3, 8, 12)},
    {'in_shape': (3, 16, 24), 'size': 24,
     'fit_short': True, 'out_shape': (3, 24, 36)},
    {'in_shape': (3, 24, 16), 'size': 36,
     'fit_short': False, 'out_shape': (3, 36, 24)},
    {'in_shape': (3, 16, 24), 'size': 36,
     'fit_short': False, 'out_shape': (3, 24, 36)},
    {'in_shape': (3, 24, 12), 'size': 12,
     'fit_short': False, 'out_shape': (3, 12, 6)},
    # grayscale
    {'in_shape': (1, 16, 24), 'size': 8,
     'fit_short': True, 'out_shape': (1, 8, 12)},
    {'in_shape': (1, 16, 24), 'size': 36,
     'fit_short': False, 'out_shape': (1, 24, 36)},
)
class TestScale(unittest.TestCase):

    def test_scale(self):
        img = np.random.uniform(size=self.in_shape)

        out = scale(img, self.size, fit_short=self.fit_short)
        self.assertEqual(out.shape, self.out_shape)


@testing.parameterize(
    {'in_shape': (3, 24, 16), 'size': 16, 'fit_short': True},
    {'in_shape': (3, 16, 24), 'size': 16, 'fit_short': True},
    {'in_shape': (3, 24, 16), 'size': 24, 'fit_short': False},
    {'in_shape': (3, 16, 24), 'size': 24, 'fit_short': False},
    # grayscale
    {'in_shape': (1, 16, 24), 'size': 24, 'fit_short': False},
)
class TestScaleNoResize(unittest.TestCase):

    def test_scale_no_resize(self):
        img = np.random.uniform(size=self.in_shape)

        out = scale(img, self.size, fit_short=self.fit_short)
        self.assertIs(img, out)


testing.run_module(__name__, __file__)
