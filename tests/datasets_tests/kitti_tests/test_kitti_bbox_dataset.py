import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import kitti_bbox_label_names
from chainercv.datasets import KITTIBboxDataset
from chainercv.utils import assert_is_bbox_dataset


@testing.parameterize(
    # category : City
    {
        'date': '2011_09_26',
        'drive_num': '0001',
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0001',
        'sync': False,
        'is_left': True,
        'tracklet': True
    },
    # {
    #     'date': '2011_09_26',
    #     'drive_num': '0001',
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': True
    # },
    {
        'date': '2011_09_26',
        'drive_num': '0001',
        'sync': True,
        'is_left': False,
        'tracklet': True
    },
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_09_26',
    #     'drive_num': '0001',
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': False
    # },
    # Test NG(Part of Framerate not Bbox/label data)
    # {
    #     'date': '2011_09_26',
    #     'drive_num': '0009',
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': True
    # },
    # Test NG(Part of Framerate not Bbox/label data)
    # {
    #     'date': '2011_09_26',
    #     'drive_num': '0017',
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': True
    # },
    {
        'date': '2011_09_26',
        'drive_num': '0056',
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0057',
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_09_28',
    #     'drive_num': '0001',
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': False
    # },
    # category : Residential
    {
        'date': '2011_09_26',
        'drive_num': '0064',
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    # category : Road
    {
        'date': '2011_09_26',
        'drive_num': '0032',
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    {
        'date': '2011_09_26',
        'drive_num': '0052',
        'sync': True,
        'is_left': True,
        'tracklet': True
    },
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_10_03',
    #     'drive_num': '0047',
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': True
    # },
    # category : Campus
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_09_28',
    #     'drive_num': '0016',
    #     'sync': True,
    #     'is_left': True,
    #     'tracklet': True
    # },
    # category : Person
    # Test NG(not Tracklet data)
    # {
    #     'date': '2011_09_28',
    #     'drive_num': '0053',
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
            sync=self.sync,
            is_left=self.is_left,
            tracklet=self.tracklet)

    @attr.slow
    def test_kitti_bbox_dataset(self):
        assert_is_bbox_dataset(
            self.dataset, len(kitti_bbox_label_names), n_example=10)


testing.run_module(__name__, __file__)
