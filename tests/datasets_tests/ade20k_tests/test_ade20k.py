import numpy as np
import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import ade20k_semantic_segmentation_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.datasets import ADE20KTestImageDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset
from chainercv.utils.testing.assertions.assert_is_image import assert_is_image


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
)
class TestADE20KSemanticSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ADE20KSemanticSegmentationDataset(split=self.split)

    @attr.slow
    def test_ade20k_dataset(self):
        assert_is_semantic_segmentation_dataset(
            self.dataset, len(ade20k_semantic_segmentation_label_names),
            n_example=10)


class TestADE20KTestImageDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ADE20KTestImageDataset()

    @attr.slow
    def test_ade20k_dataset(self):
        indices = np.random.permutation(np.arange(len(self.dataset)))
        for i in indices[:10]:
            img = self.dataset[i]
            assert_is_image(img, color=True)


testing.run_module(__name__, __file__)
