import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
    {'split': 'trainval'}
)
class TestVOCSemanticSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = VOCSemanticSegmentationDataset(split=self.split)

    @attr.slow
    def test_voc_semantic_segmentation_dataset(self):
        assert_is_semantic_segmentation_dataset(
            self.dataset,
            len(voc_semantic_segmentation_label_names),
            n_example=10)


testing.run_module(__name__, __file__)
