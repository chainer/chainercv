import unittest

import numpy as np

from chainercv.visualizations import vis_image
from chainercv.utils import testing

try:
    import matplotlib  # NOQA
    _available = True
except ImportError:
    _available = False


@testing.parameterize(
    {'img': np.random.randint(0, 255, size=(3, 32, 32)).astype(np.float32)},
    {'img': None}
)
@unittest.skipUnless(_available, 'Matplotlib is not installed')
class TestVisImage(unittest.TestCase):

    def test_vis_image(self):
        ax = vis_image(self.img)
        self.assertTrue(isinstance(ax, matplotlib.axes.Axes))


testing.run_module(__name__, __file__)
