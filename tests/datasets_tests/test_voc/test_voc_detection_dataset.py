import unittest

import numpy as np

from chainer import testing
from chainer.testing import condition
from chainer.testing import attr

from chainercv.datasets import voc_detection_label_names
from chainercv.datasets import VOCDetectionDataset


def _create_paramters():
    split_years = testing.product({
        'split': ['train', 'trainval', 'val'],
        'year': ['2007', '2012']})
    split_years += [{'split': 'test', 'year': '2007'}]
    params = testing.product_dict(
        split_years,
        [{'use_difficult': True, 'return_difficult': True},
         {'use_difficult': True, 'return_difficult': False},
         {'use_difficult': False, 'return_difficult': False}])
    return params


@testing.parameterize(*_create_paramters())
class TestVOCDetectionDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = VOCDetectionDataset(
            split=self.split,
            year=self.year,
            use_difficult=self.use_difficult,
            return_difficult=self.return_difficult)
        self.n_out = 4 if self.return_difficult else 3

    @attr.slow
    @condition.repeat(10)
    def test_get_example(self):
        i = np.random.randint(0, len(self.dataset))
        out = self.dataset[i]

        self.assertEqual(len(out), self.n_out)

        img, bbox, label = out[:3]

        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(img.shape[0], 3)
        self.assertEqual(img.ndim, 3)
        self.assertGreaterEqual(np.min(img), 0)
        self.assertLessEqual(np.max(img), 255)

        self.assertIsInstance(bbox, np.ndarray)
        self.assertEqual(bbox.dtype, np.float32)
        self.assertEqual(bbox.ndim, 2)
        self.assertEqual(bbox.shape[1], 4)

        self.assertIsInstance(label, np.ndarray)
        self.assertEqual(label.dtype, np.int32)
        self.assertEqual(label.shape, (bbox.shape[0],))
        self.assertGreaterEqual(np.min(label), 0)
        self.assertLessEqual(
            np.max(label), len(voc_detection_label_names) - 1)

        if self.n_out == 4:
            difficult = out[3]
            self.assertIsInstance(difficult, np.ndarray)
            self.assertEqual(difficult.dtype, np.bool)
            self.assertEqual(difficult.shape, (bbox.shape[0],))


testing.run_module(__name__, __file__)
