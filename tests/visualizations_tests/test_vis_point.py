import unittest

import numpy as np

from chainer import testing

from chainercv.visualizations import vis_point

try:
    import matplotlib  # NOQA
    _available = True
except ImportError:
    _available = False


@testing.parameterize(
    {'visible': np.array([[True, True, False]])},
    {'visible': None}
)
@unittest.skipUnless(_available, 'Matplotlib is not installed')
class TestVisPoint(unittest.TestCase):

    def test_vis_point(self):
        img = np.random.randint(
            0, 255, size=(3, 32, 32)).astype(np.float32)
        point = np.random.uniform(size=(1, 3, 2)).astype(np.float32)
        ax = vis_point(img, point, self.visible)

        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))


testing.run_module(__name__, __file__)
