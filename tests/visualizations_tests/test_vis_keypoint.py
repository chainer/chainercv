import unittest

import numpy as np

from chainer import testing

from chainercv.visualizations import vis_keypoint

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


@testing.parameterize(
    {'kp_mask': np.array([True, True, False])},
    {'kp_mask': None}
)
class TestVisKeypoint(unittest.TestCase):

    def test_vis_keypoint(self):
        if optional_modules:
            img = np.random.randint(
                0, 255, size=(3, 32, 32)).astype(np.float32)
            keypoint = np.random.uniform(size=(3, 2)).astype(np.float32)
            ax = vis_keypoint(img, keypoint, self.kp_mask)

            self.assertTrue(isinstance(ax, matplotlib.axes.Axes))


testing.run_module(__name__, __file__)
