import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_contain


class TestResizeContain(unittest.TestCase):

    def test_resize_contain(self):
        img = np.random.uniform(size=(3, 32, 64))
        bg_value = -1

        out, param = resize_contain(img, (96, 48), bg_value=bg_value,
                                    return_param=True)

        np.testing.assert_array_equal(img, out[:, 8:40, 16:80])
        np.testing.assert_array_equal(bg_value, out[:, 0, 0])
        self.assertEqual(param['scaled_size'], (64, 32))
        self.assertEqual(param['x_offset'], 16)
        self.assertEqual(param['y_offset'], 8)

    def test_resize_contain_canvas_small_x(self):
        img = np.random.uniform(size=(3, 32, 64))
        bg_value = -1

        out, param = resize_contain(img, (68, 16), bg_value=bg_value,
                                    return_param=True)
        self.assertEqual(param['scaled_size'], (32, 16))
        self.assertEqual(param['x_offset'], 18)
        self.assertEqual(param['y_offset'], 0)

    def test_resize_contain_canvas_small_y(self):
        img = np.random.uniform(size=(3, 32, 64))
        bg_value = -1

        out, param = resize_contain(img, (16, 24), bg_value=bg_value,
                                    return_param=True)
        self.assertEqual(param['scaled_size'], (16, 8))
        self.assertEqual(param['x_offset'], 0)
        self.assertEqual(param['y_offset'], 8)


testing.run_module(__name__, __file__)
