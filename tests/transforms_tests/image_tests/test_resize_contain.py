import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_contain


class TestResizeContain(unittest.TestCase):

    def test_resize_contain(self):
        img = np.random.uniform(size=(3, 32, 32))
        bg_value = -1

        out, param = resize_contain(img, (34, 36), bg_value=bg_value,
                                    return_param=True)

        np.testing.assert_array_equal(img, out[:, 2:34, 1:33])
        np.testing.assert_array_equal(bg_value, out[:, 0, 0])
        self.assertEqual(param['scale'], 1.)
        self.assertEqual(param['x_offset'], 1)
        self.assertEqual(param['y_offset'], 2)

    def test_resize_contain_canvas_small_x(self):
        img = np.random.uniform(size=(3, 32, 32))
        bg_value = -1

        out, param = resize_contain(img, (16, 34), bg_value=bg_value,
                                    return_param=True)
        self.assertEqual(param['scale'], 16. / 32.)
        self.assertEqual(param['x_offset'], 0)
        self.assertEqual(param['y_offset'], 9)

    def test_resize_contain_canvas_small_y(self):
        img = np.random.uniform(size=(3, 32, 32))
        bg_value = -1

        out, param = resize_contain(img, (34, 16), bg_value=bg_value,
                                    return_param=True)
        self.assertEqual(param['scale'], 16. / 32.)
        self.assertEqual(param['x_offset'], 9)
        self.assertEqual(param['y_offset'], 0)


testing.run_module(__name__, __file__)
