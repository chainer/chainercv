import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import KITTI_label_names
from chainercv.datasets import KITTIDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset


@testing.parameterize(
    {
        'date': '2011_09_26',
        'driveNo': '0001',
        'color': True,
        'sync': True,
        'isLeft': True
    },
    {
        'date': '2011_09_26',
        'driveNo': '0001',
        'color': True,
        'sync': False,
        'isLeft': True
    },
    {
        'date': '2011_09_26',
        'driveNo': '0001',
        'color': True,
        'sync': True,
        'isLeft': True
    },
    {
        'date': '2011_09_26',
        'driveNo': '0017',
        'color': True,
        'sync': True,
        'isLeft': True
    },
    {
        'date': '2011_09_28',
        'driveNo': '0001',
        'color': True,
        'sync': True,
        'isLeft': True
    },
    {
        'date': '2011_10_03',
        'driveNo': '0047',
        'color': True,
        'sync': True,
        'isLeft': True
    },
)

class TestKITTIDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = KITTIDataset(
                                    date=self.date,
                                    driveNo=self.driveNo,
                                    color=self.color,
                                    sync=self.sync,
                                    isLeft=self.isLeft)

    @attr.slow
    def test_kitti_semantic_segmentation_dataset(self):
        indices = np.random.permutation(np.arange(len(self.dataset)))
        for i in indices[:10]:
            img = self.dataset[i]
            assert_is_image(img, color=True)


testing.run_module(__name__, __file__)
