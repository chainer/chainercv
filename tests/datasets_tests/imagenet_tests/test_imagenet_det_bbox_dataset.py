import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainercv.datasets import imagenet_det_bbox_label_names
from chainercv.datasets import ImagenetDetBboxDataset
from chainercv.utils import assert_is_bbox_dataset


@testing.parameterize(
    {'split': 'train', 'return_img_label': False},
    {'split': 'train', 'return_img_label': True},
    {'split': 'val', 'return_img_label': False},
)
class TestImagenetDetBboxDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ImagenetDetBboxDataset(
            split=self.split,
            return_img_label=self.return_img_label)
        self.n_out = 5 if self.return_img_label else 3

    @attr.slow
    def test_as_bbox_dataset(self):
        assert_is_bbox_dataset(
            self.dataset, len(imagenet_det_bbox_label_names), n_example=10)

    @attr.slow
    @condition.repeat(10)
    def test_img_label(self):
        if not self.return_img_label:
            return

        i = np.random.randint(0, len(self.dataset))
        _, _, label, img_label, img_label_type = self.dataset[i]
        self.assertIsInstance(img_label, np.ndarray)
        self.assertEqual(img_label.dtype, np.int32)
        self.assertIsInstance(img_label_type, np.ndarray)
        self.assertEqual(img_label_type.dtype, np.int32)

        self.assertEqual(img_label.shape, img_label_type.shape)
        self.assertTrue(img_label.max() < len(imagenet_det_bbox_label_names)
                        and img_label.min() >= 0)
        self.assertTrue(img_label_type.max() <= 1
                        and img_label_type.min() >= -1)
        if len(label) > 0:
            pos_img_label = img_label[img_label_type >= 0]
            for lb in label:
                self.assertTrue(np.isin(lb, pos_img_label))


testing.run_module(__name__, __file__)
