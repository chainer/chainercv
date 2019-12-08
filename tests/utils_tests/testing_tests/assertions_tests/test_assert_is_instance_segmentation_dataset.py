import numpy as np
import unittest

from chainer.dataset import DatasetMixin

from chainercv.utils import assert_is_instance_segmentation_dataset
from chainercv.utils import testing


class InstanceSegmentationDataset(DatasetMixin):

    def __init__(self, *options):
        self.options = options

    def __len__(self):
        return 10

    def get_example(self, i):
        img = np.random.randint(0, 256, size=(3, 48, 64))
        n_inst = np.random.randint(10, 20)
        mask = np.random.randint(0, 2, size=(n_inst, 48, 64), dtype=np.bool)
        label = np.random.randint(0, 20, size=n_inst).astype(np.int32)

        return (img, mask, label) + self.options


class InvalidSampleSizeDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, mask, label = super(
            InvalidSampleSizeDataset, self).get_example(i)[:3]
        return img, mask


class InvalidImageDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, mask, label = super(
            InvalidImageDataset, self).get_example(i)[:3]
        return img[0], mask, label


class InvalidMaskDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, mask, label = super(
            InvalidMaskDataset, self).get_example(i)[:3]
        return img, mask[0], label


class InvalidLabelDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, mask, label = super(
            InvalidLabelDataset, self).get_example(i)[:3]
        label += 1000
        return img, mask, label


class MismatchMaskLengthDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, mask, label = super(
            MismatchMaskLengthDataset, self).get_example(i)[:3]
        return img, mask[1:], label


class MismatchLabelLengthDataset(InstanceSegmentationDataset):

    def get_example(self, i):
        img, mask, label = super(
            MismatchLabelLengthDataset, self).get_example(i)[:3]
        return img, mask, label[1:]


@testing.parameterize(
    {'dataset': InstanceSegmentationDataset(), 'valid': True},
    {'dataset': InstanceSegmentationDataset('option'), 'valid': True},
    {'dataset': InvalidSampleSizeDataset(), 'valid': False},
    {'dataset': InvalidImageDataset(), 'valid': False},
    {'dataset': InvalidMaskDataset(), 'valid': False},
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
