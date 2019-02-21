import unittest

import numpy as np

from chainer import testing

from chainercv.datasets import coco_keypoint_names
from chainercv.visualizations import vis_keypoint_coco

try:
    import matplotlib  # NOQA
    _available = True
except ImportError:
    _available = False


human_id = 0


def _generate_point(n_inst, size):
    H, W = size
    n_joint = len(coco_keypoint_names[human_id])
    ys = np.random.uniform(0, H, size=(n_inst, n_joint))
    xs = np.random.uniform(0, W, size=(n_inst, n_joint))
    point = np.stack((ys, xs), axis=2).astype(np.float32)

    visible = np.random.randint(0, 2, size=(n_inst, n_joint)).astype(np.bool)

    point_score = np.random.uniform(
        0, 6, size=(n_inst, n_joint)).astype(np.float32)
    return point, visible, point_score


@testing.parameterize(*testing.product({
    'n_inst': [3, 0],
    'use_img': [False, True],
    'use_visible': [False, True],
    'use_point_score': [False, True]
}))
@unittest.skipUnless(_available, 'matplotlib is not installed')
class TestVisKeypointCOCO(unittest.TestCase):

    def setUp(self):
        size = (32, 48)
        self.point, visible, point_score = _generate_point(self.n_inst, size)
        self.img = (np.random.randint(
            0, 255, size=(3,) + size).astype(np.float32)
            if self.use_img else None)
        self.visible = visible if self.use_visible else None
        self.point_score = point_score if self.use_point_score else None

    def test_vis_keypoint_coco(self):
        ax = vis_keypoint_coco(
            self.img, self.point, self.visible,
            self.point_score)

        self.assertIsInstance(ax, matplotlib.axes.Axes)


@unittest.skipUnless(_available, 'matplotlib is not installed')
class TestVisKeypointCOCOInvisibleInputs(unittest.TestCase):

    def setUp(self):
        size = (32, 48)
        n_inst = 10
        self.point, self.visible, self.point_score = _generate_point(
            n_inst, size)
        self.img = np.random.randint(
            0, 255, size=(3,) + size).astype(np.float32)

    def _check(self, img, point, visible, point_score):
        with self.assertRaises(ValueError):
            vis_keypoint_coco(img, point, visible, point_score)

    def test_invisible_n_inst_point(self):
        self._check(self.img, self.point[:5], self.visible, self.point_score)

    def test_invisible_n_inst_visible(self):
        self._check(self.img, self.point, self.visible[:5], self.point_score)

    def test_invisible_n_inst_point_score(self):
        self._check(self.img, self.point, self.visible, self.point_score[:5])

    def test_invisible_n_joint_point(self):
        self._check(
            self.img, self.point[:, :15], self.visible, self.point_score)

    def test_invisible_n_joint_visible(self):
        self._check(
            self.img, self.point, self.visible[:, :15], self.point_score)

    def test_invisible_n_joint_point_score(self):
        self._check(
            self.img, self.point, self.visible, self.point_score[:, :15])

    def test_invisible_visible_dtype(self):
        self._check(self.img, self.point, self.visible.astype(np.int32),
                    self.point_score)

testing.run_module(__name__, __file__)
