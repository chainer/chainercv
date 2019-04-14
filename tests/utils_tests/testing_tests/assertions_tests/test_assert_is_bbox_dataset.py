import numpy as np
import unittest

from chainer.dataset import DatasetMixin

from chainercv.utils import assert_is_bbox_dataset
from chainercv.utils import generate_random_bbox
from chainercv.utils import testing


class BboxDataset(DatasetMixin):

    def __init__(self, options=(), empty_bbox=False):
        self.options = options
        self.empty_bbox = empty_bbox

    def __len__(self):
        return 10

    def get_example(self, i):
        img = np.random.randint(0, 256, size=(3, 48, 64))
        if self.empty_bbox:
            n_bbox = 0
        else:
            n_bbox = np.random.randint(10, 20)
        bbox = generate_random_bbox(n_bbox, (48, 64), 5, 20)
        label = np.random.randint(0, 20, size=n_bbox).astype(np.int32)

        return (img, bbox, label) + self.options


class InvalidSampleSizeDataset(BboxDataset):

    def get_example(self, i):
        img, bbox, label = super(
            InvalidSampleSizeDataset, self).get_example(i)[:3]
        return img, bbox


class InvalidImageDataset(BboxDataset):

    def get_example(self, i):
        img, bbox, label = super(InvalidImageDataset, self).get_example(i)[:3]
        return img[0], bbox, label


class InvalidBboxDataset(BboxDataset):

    def get_example(self, i):
        img, bbox, label = super(InvalidBboxDataset, self).get_example(i)[:3]
        bbox += 1000
        return img, bbox, label


class InvalidLabelDataset(BboxDataset):

    def get_example(self, i):
        img, bbox, label = super(InvalidLabelDataset, self).get_example(i)[:3]
        label += 1000
        return img, bbox, label


class MismatchLengthDataset(BboxDataset):

    def get_example(self, i):
        img, bbox, label = super(
            MismatchLengthDataset, self).get_example(i)[:3]
        return img, bbox, label[1:]


@testing.parameterize(
    {'dataset': BboxDataset(), 'valid': True},
    {'dataset': BboxDataset(empty_bbox=True), 'valid': True},
    {'dataset': BboxDataset(('option',)), 'valid': True},
    {'dataset': InvalidSampleSizeDataset(), 'valid': False},
    {'dataset': InvalidImageDataset(), 'valid': False},
    {'dataset': InvalidBboxDataset(), 'valid': False},
    {'dataset': InvalidLabelDataset(), 'valid': False},
    {'dataset': MismatchLengthDataset(), 'valid': False},
)
class TestAssertIsBboxDataset(unittest.TestCase):

    def test_assert_is_bbox_dataset(self):
        if self.valid:
            assert_is_bbox_dataset(self.dataset, 20)
        else:
            with self.assertRaises(AssertionError):
                assert_is_bbox_dataset(self.dataset, 20)


testing.run_module(__name__, __file__)
