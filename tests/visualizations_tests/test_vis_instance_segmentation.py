import numpy as np
import unittest

from chainer import testing

from chainercv.visualizations import vis_instance_segmentation

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


@testing.parameterize(
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': None,
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5, 1),
        'label_names': None},
    {
        'n_bbox': 3, 'label': None, 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': None, 'score': (0, 0.5, 1),
        'label_names': None},
    {
        'n_bbox': 3, 'label': None, 'score': None,
        'label_names': None},
    {
        'n_bbox': 3, 'label': (0, 1, 1), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 0, 'label': (), 'score': (),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2'),
        'colors': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 100, 100)]},
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},
)
class TestVisInstanceSegmentation(unittest.TestCase):

    def setUp(self):
        self.img = np.random.randint(0, 255, size=(3, 32, 48))
        self.mask = np.random.randint(
            0, 2, size=(self.n_bbox, 32, 48), dtype=bool)
        if self.label is not None:
            self.label = np.array(self.label, dtype=np.int32)
        if self.score is not None:
            self.score = np.array(self.score)
        if not hasattr(self, 'colors'):
            self.colors = None

    def test_vis_instance_segmentation(self):
        if not optional_modules:
            return

        ax = vis_instance_segmentation(
            self.img, self.mask, self.label, self.score,
            label_names=self.label_names, colors=self.colors)

        self.assertIsInstance(ax, matplotlib.axes.Axes)


@testing.parameterize(
    {
        'n_bbox': 3, 'label': (0, 1), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (0, 1, 2, 1), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},

    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5, 1, 0.75),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (0, 1, 3), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},
    {
        'n_bbox': 3, 'label': (-1, 1, 2), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2')},

)
class TestVisInstanceSegmentationInvalidInputs(unittest.TestCase):

    def setUp(self):
        self.img = np.random.randint(0, 255, size=(3, 32, 48))
        self.mask = np.random.randint(
            0, 2, size=(self.n_bbox, 32, 48), dtype=bool)
        if self.label is not None:
            self.label = np.array(self.label, dtype=int)
        if self.score is not None:
            self.score = np.array(self.score)

    def test_vis_instance_segmentation_invalid_inputs(self):
        if not optional_modules:
            return

        with self.assertRaises(ValueError):
            vis_instance_segmentation(
                self.img, self.mask, self.label, self.score,
                label_names=self.label_names)


testing.run_module(__name__, __file__)
