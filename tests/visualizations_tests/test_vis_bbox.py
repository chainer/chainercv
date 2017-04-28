import unittest

import numpy as np

from chainer import testing

from chainercv.visualizations import vis_bbox

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


@testing.parameterize(
    {'label': np.array([0, 1, 1]),
     'label_names': None},
    {'label': np.array([0, 1, 2]),
     'label_names': ('class_0', 'class_1', 'class_2')}
)
class TestVisBbox(unittest.TestCase):

    def test_vis_bbox(self):
        if optional_modules:
            img = np.random.randint(
                0, 255, size=(3, 32, 32)).astype(np.float32)
            bbox = np.random.uniform(size=(3, 4)).astype(np.float32)
            ax = vis_bbox(img, bbox, self.label, self.label_names)

            self.assertTrue(isinstance(ax, matplotlib.axes.Axes))


class TestInvalidVisBbox(unittest.TestCase):

    def setUp(self):
        self.img = np.random.randint(
            0, 255, size=(3, 32, 32)).astype(np.float32)
        self.bbox = np.random.uniform(size=(3, 4)).astype(np.float32)

    def test_invalid_label_exceed(self):
        if optional_modules:
            label = np.array([0, 1, 2])
            label_names = ('class_0', 'class_1')
            with self.assertRaises(IndexError):
                vis_bbox(self.img, self.bbox, label, label_names)

    def test_invalid_length_differ(self):
        if optional_modules:
            label = np.array([0, 1])
            label_names = ('class_0', 'class_1')
            with self.assertRaises(IndexError):
                vis_bbox(self.img, self.bbox, label, label_names)


testing.run_module(__name__, __file__)
