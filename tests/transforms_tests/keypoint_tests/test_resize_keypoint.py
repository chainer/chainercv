import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize_keypoint


class TestResizeKeypoint(unittest.TestCase):

    def test_resize_keypoint(self):
        keypoint = np.random.uniform(
            low=0., high=32., size=(12, 2))

        out = resize_keypoint(keypoint, in_size=(32, 32), out_size=(64, 64))
        keypoint[:, :2] *= 2
        np.testing.assert_equal(out, keypoint)


testing.run_module(__name__, __file__)
