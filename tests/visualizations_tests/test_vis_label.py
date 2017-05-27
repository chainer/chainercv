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

    def setUp(self):
        self.label = np.random.randint(
            -1, 21, size=(48, 64)).astype(np.int32)

    def test_vis_label(self):
        if optional_modules:
            ax, legend_handles = vis_label(self.label)

            self.assertIsInstance(ax, matplotlib.axes.Axes)
            for handle in legend_handles:
                self.assertIsInstance(handle, matplotlib.patches.Patch)


testing.run_module(__name__, __file__)
