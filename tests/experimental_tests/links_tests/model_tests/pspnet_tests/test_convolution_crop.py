import numpy as np
import unittest

from chainer import testing
from chainercv.experimental.links.model.pspnet import convolution_crop


class TestConvolutionCrop(unittest.TestCase):

    def test_convolution_crop(self):
        size = (8, 6)
        stride = (8, 6)
        n_channel = 3
        img = np.random.uniform(size=(n_channel, 16, 12)).astype(np.float32)
        crop_imgs, param = convolution_crop(
            img, size, stride, return_param=True)

        self.assertEqual(crop_imgs.shape, (4, n_channel) + size)
        self.assertEqual(crop_imgs.dtype, np.float32)
        for y in range(2):
            for x in range(2):
                self.assertEqual(param['y_slices'][2 * y + x].start, 8 * y)
                self.assertEqual(
                    param['y_slices'][2 * y + x].stop, 8 * (y + 1))
                self.assertEqual(param['x_slices'][2 * y + x].start, 6 * x)
                self.assertEqual(
                    param['x_slices'][2 * y + x].stop, 6 * (x + 1))
        for i in range(4):
            self.assertEqual(param['crop_y_slices'][i].start, 0)
            self.assertEqual(param['crop_y_slices'][i].stop, 8)
            self.assertEqual(param['crop_x_slices'][i].start, 0)
            self.assertEqual(param['crop_x_slices'][i].stop, 6)


testing.run_module(__name__, __file__)
