import numpy as np
import os
import shutil
import tempfile
import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.datasets.cityscapes.cityscapes_utils import cityscapes_labels
from chainercv.datasets import CityscapesSemanticSegmentationDataset
from chainercv.datasets import CityscapesTestImageDataset
from chainercv.utils import assert_is_semantic_segmentation_dataset
from chainercv.utils.testing.assertions.assert_is_image import assert_is_image
from chainercv.utils import write_image


@testing.parameterize(
    {'split': 'train', 'n_class': 19, 'label_mode': 'fine',
     'ignore_labels': True},
    {'split': 'val', 'n_class': 34, 'label_mode': 'coarse',
     'ignore_labels': False}
)
class TestCityscapesSemanticSegmentationDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        img_dir = os.path.join(
            self.temp_dir, 'leftImg8bit/{}/aachen'.format(self.split))
        resol = 'gtFine' if self.label_mode == 'fine' else 'gtCoarse'
        label_dir = os.path.join(
            self.temp_dir, '{}/{}/aachen'.format(resol, self.split))
        os.makedirs(img_dir)
        os.makedirs(label_dir)

        for i in range(10):
            img = np.random.randint(
                0, 255, size=(3, 128, 160)).astype(np.uint8)
            write_image(img, os.path.join(
                img_dir, 'aachen_000000_0000{:02d}_leftImg8bit.png'.format(i)))

            label = np.random.randint(
                0, 34, size=(1, 128, 160)).astype(np.int32)
            write_image(label, os.path.join(
                label_dir,
                'aachen_000000_0000{:02d}_{}_labelIds.png'.format(i, resol)))

        self.dataset = CityscapesSemanticSegmentationDataset(
            self.temp_dir, self.label_mode, self.split, self.ignore_labels)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_ignore_labels(self):
        for _, label_orig in self.dataset:
            H, W = label_orig.shape
            label_out = np.ones((H, W), dtype=np.int32) * -1
            for label in cityscapes_labels:
                label_out[label_orig == label.trainId] = label.id

    @attr.slow
    def test_cityscapes_semantic_segmentation_dataset(self):
        assert_is_semantic_segmentation_dataset(
            self.dataset, self.n_class, n_example=10)


class TestCityscapesTestImageDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        img_dir = os.path.join(
            self.temp_dir, 'leftImg8bit/test/berlin')
        os.makedirs(img_dir)
        for i in range(10):
            img = np.random.randint(
                0, 255, size=(3, 128, 160)).astype(np.uint8)
            write_image(img, os.path.join(
                img_dir, 'berlin_000000_0000{:02d}_leftImg8bit.png'.format(i)))

        self.dataset = CityscapesTestImageDataset(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @attr.slow
    def test_cityscapes_dataset(self):
        indices = np.random.permutation(np.arange(len(self.dataset)))
        for i in indices[:10]:
            img = self.dataset[i]
            assert_is_image(img, color=True)


testing.run_module(__name__, __file__)
