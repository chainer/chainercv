import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize


class TestResize(unittest.TestCase):

    def test_resize_color(self):
        img = np.random.uniform(size=(3, 24, 32))
        out = resize(img, output_shape=(32, 64))
        self.assertEqual(out.shape, (3, 32, 64))

    def test_resize_grayscale(self):
        img = np.random.uniform(size=(1, 24, 32))
        out = resize(img, output_shape=(32, 64))
        self.assertEqual(out.shape, (1, 32, 64))


testing.run_module(__name__, __file__)
