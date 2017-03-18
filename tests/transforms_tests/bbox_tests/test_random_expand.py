import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import random_expand


class TestRandomExpand(unittest.TestCase):

    def test_random_expand(self):
        img = np.random.uniform(-1, 1, size=(3, 64, 32))
        bbox = np.random.uniform(0, 32, size=(10, 4))

        out_img, out_bbox = random_expand(img, bbox)

        out_img, out_bbox = random_expand(img, bbox, max_ratio=1)
        out_img, out_bbox = random_expand(img, bbox, max_ratio=2)

        out_img, out_bbox = random_expand(img, bbox, fill=128)
        out_img, out_bbox = random_expand(img, bbox, fill=(104, 117, 123))
        out_img, out_bbox = random_expand(
            img, bbox, fill=np.random.uniform(255, size=3))


testing.run_module(__name__, __file__)
