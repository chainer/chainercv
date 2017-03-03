import unittest

import numpy as np

from chainer import testing

import chainer_cv
from chainer_cv.testing import helper


class TestCropWrapper(unittest.TestCase):

    def test_crop_wrapper(self):
        array = np.random.uniform(size=(20, 3, 10, 10))
        dataset = helper.SimpleDataset(array)
        cropped_shape = (3, 5, 5)
        dataset = chainer_cv.wrappers.CropWrapper(
            dataset, [0], cropped_shape, start_idx=(0, 2, 2))

        img0 = dataset.get_example(0)

        self.assertEqual(img0.shape, cropped_shape)

        np.testing.assert_equal(img0, array[0][:, 2:2 + 5, 2: 2 + 5])


testing.run_module(__name__, __file__)
