import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import scale


@testing.parameterize(
    {'in_shape': (3, 24, 16), 'size': 8,
     'fit_short': True, 'out_shape': (3, 12, 8)},
    {'in_shape': (3, 16, 24), 'size': 8,
     'fit_short': True, 'out_shape': (3, 8, 12)},
    {'in_shape': (3, 24, 16), 'size': 36,
     'fit_short': False, 'out_shape': (3, 36, 24)},
    {'in_shape': (3, 16, 24), 'size': 36,
     'fit_short': False, 'out_shape': (3, 24, 36)}
)
class TestScale(unittest.TestCase):

    def test_scale(self):
        img = np.random.uniform(size=self.in_shape)

        out = scale(img, self.size, fit_short=self.fit_short)
        self.assertEqual(out.shape, self.out_shape)


testing.run_module(__name__, __file__)
