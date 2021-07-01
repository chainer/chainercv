import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import camvid_label_names
from chainercv.datasets import CamVidDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
    {'split': 'test'}
)
class TestCamVidDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = CamVidDataset(split=self.split)

    @attr.slow
    def test_camvid_dataset(self):
        assert_is_semantic_segmentation_dataset(
            self.dataset, len(camvid_label_names), n_example=10)


testing.run_module(__name__, __file__)
