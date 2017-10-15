import unittest

import numpy as np

from chainer import testing
from chainercv.dataset import PickableDataset


class SampleDataset(PickableDataset):
    def __init__(self, len):
        super().__init__()

        self.data_names = ('img', 'bbox', 'label', 'mask')
        self.add_getter('img', self.get_image)
        self.add_getter('mask', self.get_mask)
        self.add_getter(('bbox', 'label'), self.get_bbox_label)

        self.len = len
        self.count = 0

    def __len__(self):
        return self.len

    def get_image(self, i):
        self.count += 1
        return 'img_{:d}'.format(i)

    def get_mask(self, i):
        self.count += 1
        return 'mask_{:d}'.format(i)

    def get_bbox_label(self, i):
        self.count += 1
        return 'bbox_{:d}'.format(i), 'label_{:d}'.format(i)


class TestAnnotatedImageDatasetMixin(unittest.TestCase):

    def setUp(self):
        self.len = 10
        self.dataset = SampleDataset(self.len)

    def test_base_dataset(self):
        self.assertEqual(len(self.dataset), self.len)
        self.assertEqual(
            self.dataset[0], ('img_0', 'bbox_0', 'label_0', 'mask_0'))
        self.assertEqual(self.dataset.count, 3)

    def test_single_picked_dataset(self):
        dataset = self.dataset.pick('label')
        self.assertEqual(len(dataset), self.len)
        self.assertEqual(dataset[1], 'label_1')
        self.assertEqual(self.dataset.count, 1)

    def test_multiple_picked_dataset(self):
        dataset = self.dataset.pick('label', 'bbox')
        self.assertEqual(len(dataset), self.len)
        self.assertEqual(dataset[2], ('label_2', 'bbox_2'))
        self.assertEqual(self.dataset.count, 1)


testing.run_module(__name__, __file__)
