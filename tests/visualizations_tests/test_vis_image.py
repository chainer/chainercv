import unittest

import numpy as np

from chainer import testing

from chainercv.visualizations import vis_image

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


class TestVisImage(unittest.TestCase):

    def test_vis_image(self):
        if optional_modules:
            img = np.random.randint(
                0, 255, size=(3, 32, 32)).astype(np.float32)
            ax = vis_image(img)

            self.assertTrue(isinstance(ax, matplotlib.axes.Axes))


testing.run_module(__name__, __file__)
