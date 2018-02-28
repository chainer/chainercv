import unittest

import numpy as np

from chainer import testing

from chainercv.visualizations import vis_point

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


@testing.parameterize(
    {'mask': np.array([True, True, False])},
    {'mask': None}
)
class TestVisPoint(unittest.TestCase):

    def test_vis_point(self):
        if optional_modules:
            img = np.random.randint(
                0, 255, size=(3, 32, 32)).astype(np.float32)
            point = np.random.uniform(size=(3, 2)).astype(np.float32)
            ax = vis_point(img, point, self.mask)

            self.assertTrue(isinstance(ax, matplotlib.axes.Axes))


testing.run_module(__name__, __file__)
