import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_expand


@testing.parameterize(
    {'max_ratio': 1},
    {'max_ratio': 4},
)
class TestRandomExpand(unittest.TestCase):

    def test_random_expand(self):
        img = np.random.uniform(-1, 1, size=(3, 64, 32))

        out = random_expand(img)

        out = random_expand(img, max_ratio=1)
        np.testing.assert_equal(out, img)

        out, param = random_expand(
            img, max_ratio=self.max_ratio, return_param=True)
        ratio = param['ratio']
        y_offset = param['y_offset']
        x_offset = param['x_offset']
        np.testing.assert_equal(
            out[:, y_offset:y_offset + 64, x_offset:x_offset + 32], img)
        self.assertGreaterEqual(ratio, 1)
        self.assertLessEqual(ratio, self.max_ratio)
        self.assertEqual(out.shape[1], int(64 * ratio))
        self.assertEqual(out.shape[2], int(32 * ratio))

        out = random_expand(img, max_ratio=2)


@testing.parameterize(
    {'fill': 128},
    {'fill': (104, 117, 123)},
    {'fill':  np.random.uniform(255, size=3)},
)
class TestRandomExpandFill(unittest.TestCase):

    def test_random_expand_fill(self):
        img = np.random.uniform(-1, 1, size=(3, 64, 32))

        while True:
            out, param = random_expand(img, fill=self.fill, return_param=True)
            y_offset = param['y_offset']
            x_offset = param['x_offset']
            if y_offset > 0 or x_offset > 0:
                break

        if isinstance(self.fill, int):
            np.testing.assert_equal(
                out[:, 0, 0], (self.fill,) * 3)
        else:
            np.testing.assert_equal(
                out[:, 0, 0], self.fill)


testing.run_module(__name__, __file__)
