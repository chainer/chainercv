import unittest

import numpy as np

from chainer import testing
from chainercv.transforms import resize


class TestResizeTransform(unittest.TestCase):

    def test_resize_transform(self):
        x = np.random.uniform(size=(3, 24, 32))

        out = resize(x, output_shape=(32, 64))

        self.assertEqual(out.shape, (3, 32, 64))


testing.run_module(__name__, __file__)
