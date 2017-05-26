import numpy as np
import unittest

from chainer import testing

from chainercv.visualizations import vis_label

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


class TestVisLabel(unittest.TestCase):

    def test_vis_image(self):
        if optional_modules:
            img = np.random.randint(
                -1, 21, size=(1, 32, 32)).astype(np.int32)
            ax = vis_label(img)

            self.assertTrue(isinstance(ax, matplotlib.axes.Axes))


testing.run_module(__name__, __file__)
