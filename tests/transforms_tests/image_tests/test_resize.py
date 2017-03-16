import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize

import importlib
try:
    importlib.import_module('cv2')
    optional_modules = True
except ImportError:
    optional_modules = False


if optional_modules:
    class TestResize(unittest.TestCase):

        def test_resize(self):
            img = np.random.uniform(size=(3, 24, 32))

            out = resize(img, output_shape=(32, 64))

            self.assertEqual(out.shape, (3, 32, 64))


testing.run_module(__name__, __file__)
