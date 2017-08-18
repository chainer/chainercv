import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import cub_label_names
from chainercv.datasets import CUBLabelDataset
from chainercv.utils import assert_is_classification_dataset


@testing.parameterize(
    {'crop_bbox': True},
    {'crop_bbox': False}
)
class TestCUBLabelDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = CUBLabelDataset(crop_bbox=self.crop_bbox)

    @attr.slow
    def test_cub_label_dataset(self):
        assert_is_classification_dataset(
            self.dataset, len(cub_label_names), n_example=10)


testing.run_module(__name__, __file__)
