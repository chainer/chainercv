import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import chw_to_pil_image
from chainercv.transforms import chw_to_pil_image_tuple


class TestCHWToPILImageTransform(unittest.TestCase):

    def test_chw_to_pil_image_transform(self):
        img_uint8 = np.random.uniform(size=(3, 24, 24)).astype(np.uint8)
        img = img_uint8.astype(np.float32)

        img_obtained = chw_to_pil_image(img, reverse_color_channel=False)
        np.testing.assert_equal(img_obtained.transpose(2, 0, 1), img_uint8)
        img_obtained = chw_to_pil_image(img, reverse_color_channel=True)
        np.testing.assert_equal(
            img_obtained.transpose(2, 0, 1), img_uint8[::-1, :, :])

    def test_chw_to_pil_image_tuple_transform(self):
        img_uint8 = np.random.uniform(size=(3, 24, 24)).astype(np.uint8)
        img = img_uint8.astype(np.float32)
        y_uint8 = np.random.uniform(size=(3, 24, 24)).astype(np.uint8)
        y = y_uint8.astype(np.float32)
        imgs = (img, y)

        img_obtained, y_obtained = chw_to_pil_image_tuple(
            imgs, indices=[0], reverse_color_channel=False)
        np.testing.assert_equal(img_obtained.transpose(2, 0, 1), img_uint8)
        np.testing.assert_equal(y_obtained, y)

        img_obtained, y_obtained = chw_to_pil_image_tuple(
            imgs, indices=[0, 1], reverse_color_channel=False)
        np.testing.assert_equal(img_obtained.transpose(2, 0, 1), img_uint8)
        np.testing.assert_equal(y_obtained.transpose(2, 0, 1), y)

        img_obtained, y_obtained = chw_to_pil_image_tuple(
            imgs, indices=[0, 1], reverse_color_channel=True)
        np.testing.assert_equal(
            img_obtained.transpose(2, 0, 1), img_uint8[::-1, :, :])
        np.testing.assert_equal(
            y_obtained.transpose(2, 0, 1), y_uint8[::-1, :, :])


testing.run_module(__name__, __file__)
