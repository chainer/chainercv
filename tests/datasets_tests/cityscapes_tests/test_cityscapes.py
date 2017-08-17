import os
import shutil
import tempfile
import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr
from chainercv.datasets.cityscapes.cityscapes_utils import cityscapes_labels
from chainercv.datasets import cityscapes_label_names
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset
from chainercv.utils import write_image


@testing.parameterize(
    {'split': 'train', 'ignore_labels': True},
    {'split': 'val', 'ignore_labels': False}
)
class TestCityscapesSemanticSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        img_dir = os.path.join(
            self.temp_dir, 'leftImg8bit/{}/aachen'.format(self.split))
        label_dir = os.path.join(
            self.temp_dir, 'gtFine/{}/aachen'.format(self.split))
        os.makedirs(img_dir)
        os.makedirs(label_dir)

        for i in range(10):
            img = np.random.randint(
                0, 255, size=(128, 160, 3)).astype(np.uint8)
            write_image(img, os.path.join(
                img_dir, 'aachen_000000_0000{:02d}_leftImg8bit.png'.format(i)))

            label = np.random.randint(-1, 33, size=(128, 160)).astype(np.uint8)
            label.save(label, os.path.join(
                label_dir,
                'aachen_000000_0000{:02d}_gtFine_labelIds.png'.format(i)))

        img_dir = os.path.join(self.temp_dir, 'leftImg8bit')
        label_dir = os.path.join(self.temp_dir, 'gtFine')
        self.dataset = CityscapesSemanticSegmentationDataset(
            img_dir, label_dir, self.split, self.ignore_labels)

    def test_ignore_labels(self):
        _, label_orig = self.dataset
        H, W = label_orig.shape
        label_out = np.ones((H, W), dtype=np.int32) * -1
        for label in cityscapes_labels:
            label_out[label_orig == label.trainId] = label.id

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @attr.slow
    def test_cityscapes_semantic_segmentation_dataset(self):
        assert_is_semantic_segmentation_dataset(
            self.dataset, len(cityscapes_label_names), n_example=10)


testing.run_module(__name__, __file__)
