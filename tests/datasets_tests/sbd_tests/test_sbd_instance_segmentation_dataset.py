import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import sbd_instance_segmentation_label_names
from chainercv.datasets import SBDInstanceSegmentationDataset
from chainercv.utils import assert_is_instance_segmentation_dataset

try:
    import scipy  # NOQA
    _available = True
except ImportError:
    _available = False


@testing.parameterize(
    {'split': 'train'},
    {'split': 'val'},
    {'split': 'trainval'}
)
@unittest.skipUnless(_available, 'SciPy is not installed')
class TestSBDInstanceSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = SBDInstanceSegmentationDataset(split=self.split)

    @attr.slow
    def test_sbd_instance_segmentation_dataset(self):
        assert_is_instance_segmentation_dataset(
            self.dataset,
            len(sbd_instance_segmentation_label_names),
            n_example=10)


testing.run_module(__name__, __file__)
