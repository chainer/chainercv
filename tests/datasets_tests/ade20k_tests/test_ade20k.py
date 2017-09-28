import unittest

from chainer import testing
from chainer.testing import attr
import numpy as np

from chainercv.datasets import ade20k_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import ADE20KTestImageDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset
from chainercv.utils.testing.assertions.assert_is_image import assert_is_image


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
    {'split': 'test'}
)
class TestADE20KDataset(unittest.TestCase):

    def setUp(self):
        if self.split == 'train' or self.split == 'val':
            self.dataset = ADE20KSemanticSegmentationDataset(split=self.split)
        else:
            self.dataset = ADE20KTestImageDataset()

    @attr.slow
    def test_ade20k_dataset(self):
        if self.split == 'train' or self.split == 'val':
            assert_is_semantic_segmentation_dataset(
                self.dataset, len(ade20k_label_names), n_example=10)
        else:
            idx = np.random.permutation(np.arange(len(self.dataset)))
            for i in idx[:10]:
                img = self.dataset[i]
                assert_is_image(img, color=True)


testing.run_module(__name__, __file__)
