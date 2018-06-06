import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import kitti_bbox_label_names
from chainercv.datasets import KITTIBboxDataset
from chainercv.utils import assert_is_bbox_dataset


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
class TestKITTIBboxDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = KITTIBboxDataset(
            date=self.date,
            driveNo=self.driveNo,
            color=self.color,
            sync=self.sync,
            isLeft=self.isLeft)

    @attr.slow
    def test_kitti_bbox_dataset(self):
        assert_is_bbox_dataset(
            self.dataset, len(kitti_bbox_label_names), n_example=10)


testing.run_module(__name__, __file__)
