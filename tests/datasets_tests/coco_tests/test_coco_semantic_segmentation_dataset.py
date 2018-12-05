import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import coco_semantic_segmentation_label_names
from chainercv.datasets import COCOSemanticSegmentationDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
)
class TestCOCOSemanticSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = COCOSemanticSegmentationDataset(split=self.split)

    @attr.slow
    def test_coco_semantic_segmentation_dataset(self):
        assert_is_semantic_segmentation_dataset(
            self.dataset,
            len(coco_semantic_segmentation_label_names),
            n_example=10)


testing.run_module(__name__, __file__)
