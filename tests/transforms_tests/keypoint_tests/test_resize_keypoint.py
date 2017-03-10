import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_keypoint


class TestRandomCropTransform(unittest.TestCase):

    def test_random_crop(self):
        keypoint = np.random.uniform(
            low=0., high=32., size=(12, 3))

        out = resize_keypoint(
            keypoint, input_shape=(32, 32),
            output_shape=(64, 64))
        keypoint[:, :2] *= 2
        np.testing.assert_equal(out, keypoint)


testing.run_module(__name__, __file__)
