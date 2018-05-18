import unittest

import numpy as np

from chainer import testing

from chainercv.utils import generate_random_bbox
from chainercv.visualizations import vis_bbox

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
        'label_names': ('c0', 'c1', 'c2'), 'no_img': True},
    {
        'n_bbox': 3, 'label': (0, 1, 2), 'score': (0, 0.5, 1),
        'label_names': ('c0', 'c1', 'c2'),
        'instance_colors': [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 100, 100)]},
)
class TestVisBbox(unittest.TestCase):

    def setUp(self):
        if hasattr(self, 'no_img'):
            self.img = None
        else:
            self.img = np.random.randint(0, 255, size=(3, 32, 48))
        self.bbox = generate_random_bbox(
            self.n_bbox, (48, 32), 8, 16)
        if self.label is not None:
            self.label = np.array(self.label, dtype=int)
        if self.score is not None:
            self.score = np.array(self.score)
        if not hasattr(self, 'instance_colors'):
            self.instance_colors = None

    def test_vis_bbox(self):
        if not optional_modules:
            return

        ax = vis_bbox(
            self.img, self.bbox, self.label, self.score,
            label_names=self.label_names,
            instance_colors=self.instance_colors)

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
class TestVisBboxInvalidInputs(unittest.TestCase):

    def setUp(self):
        self.img = np.random.randint(0, 255, size=(3, 32, 48))
        self.bbox = np.random.uniform(size=(self.n_bbox, 4))
        if self.label is not None:
            self.label = np.array(self.label, dtype=int)
        if self.score is not None:
            self.score = np.array(self.score)
        if not hasattr(self, 'instance_colors'):
            self.instance_colors = None

    def test_vis_bbox_invalid_inputs(self):
        if not optional_modules:
            return

        with self.assertRaises(ValueError):
            vis_bbox(
                self.img, self.bbox, self.label, self.score,
                label_names=self.label_names,
                instance_colors=self.instance_colors)


testing.run_module(__name__, __file__)
