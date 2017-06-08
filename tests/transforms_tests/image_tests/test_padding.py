import unittest

import numpy as np

from chainer import testing

from chainercv.transforms import padding


class TestPadding(unittest.TestCase):

    def test_padding(self):
        img = np.random.uniform(-1, 1, size=(3, 64, 32))

        out = padding(img, (64, 72))
        self.assertEqual(out.shape, (3, 72, 64))

        out, param = padding(
            img, (64, 72), return_param=True)
        x_offset = param['x_offset']
        y_offset = param['y_offset']
        self.assertEqual(x_offset, 16)
        self.assertEqual(y_offset, 4)
        np.testing.assert_equal(
            out[:, y_offset:y_offset + 64, x_offset:x_offset + 32], img)

    def test_padding_invalid_size(self):
        img = np.random.uniform(-1, 1, size=(3, 64, 32))

        with self.assertRaises(ValueError):
            padding(img, (64, 60))

        with self.assertRaises(ValueError):
            padding(img, (30, 72))


@testing.parameterize(
    {'fill': 128},
    {'fill': (104, 117, 123)},
    {'fill':  np.random.uniform(255, size=3)},
)
class TestPaddingFill(unittest.TestCase):

    def test_padding_fill(self):
        img = np.random.uniform(-1, 1, size=(3, 64, 32))

        out = padding(img, (72, 64), fill=self.fill)

        if isinstance(self.fill, int):
            np.testing.assert_equal(
                out[:, 0, 0], (self.fill,) * 3)
        else:
            np.testing.assert_equal(
                out[:, 0, 0], self.fill)


testing.run_module(__name__, __file__)
