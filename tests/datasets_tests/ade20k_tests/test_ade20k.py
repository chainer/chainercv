import unittest

from chainer import testing
from chainer.testing import attr
from chainercv.datasets import ade20k_label_names
from chainercv.datasets import ADE20KSemanticSegmentationDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset
from chainercv.utils.testing.assertions.assert_is_image import assert_is_image


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
    {'split': 'test'}
)
class TestADE20KDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ADE20KSemanticSegmentationDataset(split=self.split)

    @attr.slow
    def test_camvid_dataset(self):
        if self.split == 'train' or self.split == 'val':
            assert_is_semantic_segmentation_dataset(
                self.dataset, len(ade20k_label_names), n_example=10)
        else:
            for img in self.dataset[:10]:
                assert_is_image(img, color=True)


testing.run_module(__name__, __file__)
