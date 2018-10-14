import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import voc_instance_segmentation_label_names
from chainercv.datasets import VOCInstanceSegmentationDataset
from chainercv.utils import assert_is_instance_segmentation_dataset


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
    {'split': 'trainval'}
)
class TestVOCInstanceSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = VOCInstanceSegmentationDataset(split=self.split)

    @attr.slow
    def test_voc_instance_segmentation_dataset(self):
        assert_is_instance_segmentation_dataset(
            self.dataset,
            len(voc_instance_segmentation_label_names),
            n_example=10)


testing.run_module(__name__, __file__)
