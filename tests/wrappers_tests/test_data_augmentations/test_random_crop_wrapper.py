import unittest

from chainer import testing

import chainer_cv
from chainer_cv.testing import helper


class TestRandomCropWrapper(unittest.TestCase):

    def test_random_crop_wrapper(self):
        dataset = helper.DummyDataset(shapes=[(3, 10, 10), (3, 10, 10)])
        cropped_shape = (3, 5, 5)
        dataset = chainer_cv.wrappers.RandomCropWrapper(
            dataset, [0], cropped_shape)

        img0, img1 = dataset.get_example(0)

        self.assertEqual(img0.shape, cropped_shape)
        self.assertEqual(img1.shape, (3, 10, 10))


testing.run_module(__name__, __file__)
