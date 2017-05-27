import numpy as np
import unittest

from chainer import testing

from chainercv.visualizations import vis_label

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


@testing.parameterize(*testing.product({
    'label_names': [None, ('class0', 'class1', 'class2')],
    'label_colors': [None, ((255, 0, 0), (0, 255, 0), (0, 0, 255))],
}))
class TestVisLabel(unittest.TestCase):

    def setUp(self):
        self.label = np.random.randint(
            -1, 3, size=(48, 64)).astype(np.int32)

    def test_vis_label(self):
        if optional_modules:
            ax, legend_handles = vis_label(
                self.label,
                label_names=self.label_names, label_colors=self.label_colors)

            self.assertIsInstance(ax, matplotlib.axes.Axes)
            for handle in legend_handles:
                self.assertIsInstance(handle, matplotlib.patches.Patch)


class TestVisLabelInvalidArguments(unittest.TestCase):

    def test_vis_label_mismatch_names_and_colors(self):
        label = np.random.randint(-1, 2, size=(48, 64)).astype(np.int32)

        if optional_modules:
            with self.assertRaises(ValueError):
                vis_label(
                    label,
                    label_names=('class0', 'class1', 'class2'),
                    label_colors=((255, 0, 0), (0, 255, 0)))

    def test_vis_label_exceed_value(self):
        label = np.random.randint(10, 20, size=(48, 64)).astype(np.int32)

        if optional_modules:
            with self.assertRaises(ValueError):
                vis_label(
                    label,
                    label_names=('class0', 'class1', 'class2'))


testing.run_module(__name__, __file__)
