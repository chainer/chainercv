import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
    {'split': 'test'}
)
class TestCamVidDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = CamVidDataset(split=self.split)

    @attr.slow
    @condition.repeat(10)
    def test_camvid_dataset(self):
        i = np.random.randint(0, len(self.dataset))

        img, label = self.dataset[i]

        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(img.shape[0], 3)
        self.assertGreaterEqual(np.min(img), 0)
        self.assertLessEqual(np.max(img), 255)

        self.assertIsInstance(label, np.ndarray)
        self.assertEqual(label.dtype, np.int32)
        self.assertEqual(label.shape, img.shape[1:])
        self.assertGreaterEqual(np.min(label), -1)
        self.assertLessEqual(
            np.max(label), len(camvid_label_names) - 1)


testing.run_module(__name__, __file__)
