import numpy as np
import unittest

from chainer import testing

from chainercv.utils import generate_random_bbox


class TestGenerateRandomBbox(unittest.TestCase):

    def test_generate_random_bbox(self):
        n = 32
        img_size = (128, 256)
        min_length = 16
        max_length = 48

        bbox = generate_random_bbox(n, img_size, min_length, max_length)

        self.assertIsInstance(bbox, np.ndarray)
        self.assertEqual(bbox.shape, (n, 4))
        self.assertTrue(np.all(bbox[:, [0, 2]] < img_size[0]))
        self.assertTrue(np.all(bbox[:, [0, 2]] >= 0))
        self.assertTrue(np.all(bbox[:, [1, 3]] < img_size[1]))
        self.assertTrue(np.all(bbox[:, [1, 3]] >= 0))

        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 3] - bbox[:, 1]
        self.assertTrue(np.all(w < max_length))
        self.assertTrue(np.all(w >= min_length))
        self.assertTrue(np.all(h < max_length))
        self.assertTrue(np.all(h >= min_length))


testing.run_module(__name__, __file__)
