import numpy as np
import unittest

from chainer.dataset import DatasetMixin
from chainer import testing

from chainercv.utils import assert_is_instance_segmentation_dataset
from chainercv.utils import generate_random_bbox


class InstanceSegmentationDataset(DatasetMixin):

    def __init__(self, *options):
        self.options = options

    def __len__(self):
        return 10

    def get_example(self, i):
        img = np.random.randint(0, 256, size=(3, 48, 64))
        n_bbox = np.random.randint(10, 20)
        mask = np.random.randint(0, 2, size=(n_bbox, 48, 64), dtype=np.bool)
        bbox = generate_random_bbox(n_bbox, (48, 64), 5, 20)
        label = np.random.randint(0, 20, size=n_bbox).astype(np.int32)

        return (img, bbox, mask, label) + self.options


class InvalidSampleSizeDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, bbox, mask, label = super(
            InvalidSampleSizeDataset, self).get_example(i)[:4]
        return img, bbox, mask


class InvalidImageDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, bbox, mask, label = super(
            InvalidImageDataset, self).get_example(i)[:4]
        return img[0], bbox, mask, label


class InvalidMaskDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, bbox, mask, label = super(
            InvalidMaskDataset, self).get_example(i)[:4]
        return img, bbox, mask[0], label


class InvalidBboxDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, bbox, mask, label = super(
            InvalidBboxDataset, self).get_example(i)[:4]
        bbox += 1000
        return img, bbox, mask, label


class InvalidLabelDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, bbox, mask, label = super(
            InvalidLabelDataset, self).get_example(i)[:4]
        label += 1000
        return img, bbox, mask, label


class MismatchMaskLengthDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, bbox, mask, label = super(
            MismatchMaskLengthDataset, self).get_example(i)[:4]
        return img, bbox, mask[1:], label


class MismatchLabelLengthDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, bbox, mask, label = super(
            MismatchLabelLengthDataset, self).get_example(i)[:4]
        return img, bbox, mask, label[1:]


@testing.parameterize(
    {'dataset': InstanceSegmentationDataset(), 'valid': True},
    {'dataset': InstanceSegmentationDataset('option'), 'valid': True},
    {'dataset': InvalidSampleSizeDataset(), 'valid': False},
    {'dataset': InvalidImageDataset(), 'valid': False},
    {'dataset': InvalidMaskDataset(), 'valid': False},
    {'dataset': InvalidBboxDataset(), 'valid': False},
    {'dataset': InvalidLabelDataset(), 'valid': False},
    {'dataset': MismatchMaskLengthDataset(), 'valid': False},
    {'dataset': MismatchLabelLengthDataset(), 'valid': False},
)
class TestAssertIsSemanticSegmentationDataset(unittest.TestCase):

    def test_assert_is_semantic_segmentation_dataset(self):
        if self.valid:
            assert_is_instance_segmentation_dataset(self.dataset, 20)
        else:
            with self.assertRaises(AssertionError):
                assert_is_instance_segmentation_dataset(self.dataset, 20)


testing.run_module(__name__, __file__)
