import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import flip_keypoint


class TestFlipKeypoint(unittest.TestCase):

    def test_flip_keypoint(self):
        keypoint = np.random.uniform(
            low=0., high=32., size=(12, 2))

        out = flip_keypoint(keypoint, size=(34, 32), y_flip=True)
        keypoint_expected = keypoint.copy()
        keypoint_expected[:, 0] = 33 - keypoint[:, 0]
        np.testing.assert_equal(out, keypoint_expected)

        out = flip_keypoint(keypoint, size=(34, 32), x_flip=True)
        keypoint_expected = keypoint.copy()
        keypoint_expected[:, 1] = 31 - keypoint[:, 1]
        np.testing.assert_equal(out, keypoint_expected)


testing.run_module(__name__, __file__)
