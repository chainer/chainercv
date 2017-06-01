import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import translate_keypoint


class TestTranslateKeypoint(unittest.TestCase):

    def test_translate_keypoint(self):
        keypoint = np.random.uniform(
            low=0., high=32., size=(10, 2))

        out = translate_keypoint(keypoint, y_offset=3, x_offset=5)
        expected = np.empty_like(keypoint)
        expected[:, 0] = keypoint[:, 0] + 3
        expected[:, 1] = keypoint[:, 1] + 5
        np.testing.assert_equal(out, expected)


testing.run_module(__name__, __file__)
