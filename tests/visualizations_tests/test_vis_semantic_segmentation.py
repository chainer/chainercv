import numpy as np
import unittest

from chainer import testing

from chainercv.visualizations import vis_semantic_segmentation

try:
    import matplotlib  # NOQA
    _available = True
except ImportError:
    _available = False


@testing.parameterize(*testing.product({
    'label_names': [None, ('class0', 'class1', 'class2')],
    'label_colors': [None, ((255, 0, 0), (0, 255, 0), (0, 0, 255))],
    'all_label_names_in_legend': [False, True],
    'no_img': [False, True],
}))
@unittest.skipUnless(_available, 'Matplotlib is not installed')
class TestVisSemanticSegmentation(unittest.TestCase):

    def setUp(self):
        if self.no_img:
            self.img = None
        else:
            self.img = np.random.randint(0, 255, size=(3, 32, 48))
        self.label = np.random.randint(
            -1, 3, size=(48, 64)).astype(np.int32)

    def test_vis_semantic_segmentation(self):
        ax, legend_handles = vis_semantic_segmentation(
            self.img, self.label,
            label_names=self.label_names, label_colors=self.label_colors,
            all_label_names_in_legend=self.all_label_names_in_legend)

        self.assertIsInstance(ax, matplotlib.axes.Axes)
        for handle in legend_handles:
            self.assertIsInstance(handle, matplotlib.patches.Patch)


@unittest.skipUnless(_available, 'Matplotlib is not installed')
class TestVisSemanticSegmentationInvalidArguments(unittest.TestCase):

    def test_vis_semantic_segmentation_mismatch_names_and_colors(self):
        label = np.random.randint(-1, 2, size=(48, 64)).astype(np.int32)
        with self.assertRaises(ValueError):
            vis_semantic_segmentation(
                None, label,
                label_names=('class0', 'class1', 'class2'),
                label_colors=((255, 0, 0), (0, 255, 0)))

    def test_vis_semantic_segmentation_exceed_value(self):
        label = np.random.randint(10, 20, size=(48, 64)).astype(np.int32)
        with self.assertRaises(ValueError):
            vis_semantic_segmentation(
                None, label,
                label_names=('class0', 'class1', 'class2'))


testing.run_module(__name__, __file__)
