import numpy as np
import unittest

from chainer.dataset import DatasetMixin
from chainer import testing

from chainercv.utils import assert_is_semantic_segmentation_dataset


class SemanticSegmentationDataset(DatasetMixin):

    def __init__(self, *options):
        self.options = options

    def __len__(self):
        return 10

    def get_example(self, i):
        img = np.random.randint(0, 256, size=(3, 48, 64))
        label = np.random.randint(-1, 21, size=(48, 64)).astype(np.int32)

        return (img, label) + self.options


class InvalidSampleSizeDataset(SemanticSegmentationDataset):

    def get_example(self, i):
        img, label = super(
            InvalidSampleSizeDataset, self).get_example(i)[:2]
        return img


class InvalidImageDataset(SemanticSegmentationDataset):

    def get_example(self, i):
        img, label = super(InvalidImageDataset, self).get_example(i)[:2]
        return img[0], label


class InvalidLabelDataset(SemanticSegmentationDataset):

    def get_example(self, i):
        img, label = super(InvalidLabelDataset, self).get_example(i)[:2]
        label += 1000
        return img, label


@testing.parameterize(
    {'dataset': SemanticSegmentationDataset(), 'valid': True},
    {'dataset': SemanticSegmentationDataset('option'), 'valid': True},
    {'dataset': InvalidSampleSizeDataset(), 'valid': False},
    {'dataset': InvalidImageDataset(), 'valid': False},
    {'dataset': InvalidLabelDataset(), 'valid': False},
)
class TestAssertIsSemanticSegmentationDataset(unittest.TestCase):

    def test_assert_is_semantic_segmentation_dataset(self):
        if self.valid:
            assert_is_semantic_segmentation_dataset(self.dataset, 21)
        else:
            with self.assertRaises(AssertionError):
                assert_is_semantic_segmentation_dataset(self.dataset, 21)


testing.run_module(__name__, __file__)
