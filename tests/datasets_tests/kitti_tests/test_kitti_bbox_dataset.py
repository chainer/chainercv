import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import kitti_bbox_label_names
from chainercv.datasets import KITTIBboxDataset
from chainercv.utils import assert_is_bbox_dataset


@testing.parameterize(
    {
        'date': '2011_09_26',
        'drive_num': '0001',
        'color': True,
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0001',
        'color': True,
        'sync': False,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0001',
        'color': False,
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0001',
        'color': True,
        'sync': True,
        'is_left': False,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0005',
        'color': True,
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0056',
        'color': True,
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0057',
        'color': True,
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_09_26',
    #     'drive_num': '0001',
    #     'color': True,
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': False
    # },
    # Test NG(Part of Framerate not Bbox/label data)
    # {
    #     'date': '2011_09_26',
    #     'drive_num': '0017',
    #     'color': True,
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': True
    # },
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_09_28',
    #     'drive_num': '0001',
    #     'color': True,
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': False
    # },
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_10_03',
    #     'drive_num': '0047',
    #     'color': True,
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': True
    # },
)
class TestKITTIBboxDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = KITTIBboxDataset(
            date=self.date,
            drive_num=self.drive_num,
            color=self.color,
            sync=self.sync,
            is_left=self.is_left,
            tracklet=self.tracklet)

    @attr.slow
    def test_kitti_bbox_dataset(self):
        assert_is_bbox_dataset(
            self.dataset, len(kitti_bbox_label_names), n_example=10)


testing.run_module(__name__, __file__)
