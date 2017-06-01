import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_contain


@testing.parameterize(
    {'fill': 128},
    {'fill': (104, 117, 123)},
    {'fill':  np.random.uniform(255, size=3)},
)
class TestResizeContain(unittest.TestCase):

    def test_resize_contain(self):
        img = np.random.uniform(size=(3, 32, 64))

        out, param = resize_contain(
            img, (48, 96), fill=self.fill, return_param=True)

        np.testing.assert_array_equal(img, out[:, 8:40, 16:80])
        np.testing.assert_array_equal(self.fill, out[:, 0, 0])
        self.assertEqual(param['scaled_size'], (32, 64))
        self.assertEqual(param['y_offset'], 8)
        self.assertEqual(param['x_offset'], 16)

    def test_resize_contain_canvas_small_x(self):
        img = np.random.uniform(size=(3, 32, 64))

        out, param = resize_contain(
            img, (16, 68), fill=self.fill, return_param=True)
        self.assertEqual(param['scaled_size'], (16, 32))
        self.assertEqual(param['y_offset'], 0)
        self.assertEqual(param['x_offset'], 18)

    def test_resize_contain_canvas_small_y(self):
        img = np.random.uniform(size=(3, 32, 64))

        out, param = resize_contain(
            img, (24, 16), fill=self.fill, return_param=True)
        self.assertEqual(param['scaled_size'], (8, 16))
        self.assertEqual(param['y_offset'], 8)
        self.assertEqual(param['x_offset'], 0)


testing.run_module(__name__, __file__)
