import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import color_jitter


class TestColorJitter(unittest.TestCase):

    def test_color_jitter_run_data_augmentation(self):
        img = 255 * np.random.uniform(size=(3, 48, 32)).astype(np.float32)

        out, param = color_jitter(img, return_param=True)
        self.assertEqual(out.shape, (3, 48, 32))
        self.assertEqual(out.dtype, img.dtype)
        self.assertLessEqual(np.max(img), 255)
        self.assertGreaterEqual(np.min(img), 0)

        self.assertEqual(
            sorted(param['order']), ['brightness', 'contrast', 'saturation'])
        self.assertIsInstance(param['brightness_alpha'], float)
        self.assertIsInstance(param['contrast_alpha'], float)
        self.assertIsInstance(param['saturation_alpha'], float)

    def test_color_jitter_no_data_augmentation(self):
        img = 255 * np.random.uniform(size=(3, 48, 32)).astype(np.float32)

        out, param = color_jitter(img, 0, 0, 0, return_param=True)
        np.testing.assert_equal(out, img)
        self.assertEqual(param['brightness_alpha'], 1)
        self.assertEqual(param['contrast_alpha'], 1)
        self.assertEqual(param['saturation_alpha'], 1)


testing.run_module(__name__, __file__)
