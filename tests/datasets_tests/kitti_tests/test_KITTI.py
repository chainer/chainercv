import numpy as np
import os
import shutil
import tempfile
import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import KITTIDataset
from chainercv.datasets.kitti.kitti_utils import kitti_labels
from chainercv.utils import assert_is_semantic_segmentation_dataset
from chainercv.utils.testing.assertions.assert_is_image import assert_is_image
from chainercv.utils import write_image


@testing.parameterize(
    {
        'date': '2011_09_26', 
        'driveNo': '0001', 
        'imgNo': '00',
        'sync': True
    },
    {
        'date': '2011_09_26', 
        'driveNo': '0001', 
        'imgNo': '00',
        'sync': False
    },
    {
        'date': '2011_09_26', 
        'driveNo': '0001', 
        'imgNo': '02',
        'sync': True
    },
    {
        'date': '2011_09_26', 
        'driveNo': '0017', 
        'imgNo': '00',
        'sync': True
    },
    {
        'date': '2011_09_28', 
        'driveNo': '0001', 
        'imgNo': '00',
        'sync': True
    },
    {
        'date': '2011_10_03', 
        'driveNo': '0047', 
        'imgNo': '00',
        'sync': True
    },
)

class TestKITTIDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = KITTIDataset(
                                    date=self.date,
                                    driveNo=self.driveNo,
                                    imgNo=self.imgNo, 
                                    sync=self.sync)


    @attr.slow
    @attr.disk
    def test_kitti_semantic_segmentation_dataset(self):
        indices = np.random.permutation(np.arange(len(self.dataset)))
        for i in indices[:10]:
            img = self.dataset[i]
            assert_is_image(img, color=True)


testing.run_module(__name__, __file__)
