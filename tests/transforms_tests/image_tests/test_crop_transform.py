import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_crop


class TestRandomCropTransform(unittest.TestCase):

    def test_random_crop_transform(self):
        x = np.random.uniform(size=(3, 32, 32))

        out = random_crop(x, (None, 32, 32))
        np.testing.assert_equal(out, x)


testing.run_module(__name__, __file__)
