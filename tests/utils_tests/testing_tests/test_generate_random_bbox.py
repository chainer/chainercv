import numpy as np
import unittest

from chainer import testing

from chainercv.utils import assert_is_bbox
from chainercv.utils import generate_random_bbox


@testing.parameterize(
    {'n': 0},
    {'n': 32}
)
class TestGenerateRandomBbox(unittest.TestCase):

    def test_generate_random_bbox(self):
        img_size = (128, 256)
        min_length = 16
        max_length = 48

        bbox = generate_random_bbox(self.n, img_size, min_length, max_length)

        assert_is_bbox(bbox, img_size)
        self.assertEqual(bbox.shape[0], self.n)

        if self.n > 0:
            h = bbox[:, 2] - bbox[:, 0]
            w = bbox[:, 3] - bbox[:, 1]
            self.assertTrue(np.all(h < max_length))
            self.assertTrue(np.all(h >= min_length))
            self.assertTrue(np.all(w < max_length))
            self.assertTrue(np.all(w >= min_length))


testing.run_module(__name__, __file__)
