import unittest

import numpy as np
import os
from PIL import Image
import tempfile

from chainer import testing

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingClassificationDataset
from chainercv.utils import assert_is_classification_dataset


def _save_img_file(path, size, color):
    if color:
        img = np.random.randint(
            0, 255, size=(3,) + size, dtype=np.uint8)
        Image.fromarray(img.transpose(1, 2, 0)).save(path)
    else:
        img = np.random.randint(
            0, 255, size=(1,) + size, dtype=np.uint8)
        Image.fromarray(img[0]).save(path)


def _setup_depth_one_dummy_data(tmp_dir, n_class, n_img_per_class,
                                size, color, suffix):
    for i in range(n_class):
        class_dir = os.path.join(tmp_dir, 'class_{}'.format(i))
        os.makedirs(class_dir)
        for j in range(n_img_per_class):
            filename = os.path.join(class_dir, 'img{}.{}'.format(j, suffix))
            _save_img_file(filename, size, color)
        open(os.path.join(class_dir, 'dummy_file.XXX'), 'a').close()


def _setup_depth_two_dummy_data(tmp_dir, n_class, n_img_per_class,
                                size, color, suffix):
    for i in range(n_class):
        class_dir = os.path.join(tmp_dir, 'class_{}'.format(i))
        os.makedirs(class_dir)
        nested_dir = os.path.join(class_dir, 'nested_directory')
        os.makedirs(nested_dir)
        for j in range(n_img_per_class):
            filename = os.path.join(nested_dir, 'img{}.{}'.format(j, suffix))
            _save_img_file(filename, size, color)
        open(os.path.join(nested_dir, 'dummy_file.XXX'), 'a').close()


@testing.parameterize(*testing.product({
    'size': [(48, 32)],
    'color': [True, False],
    'n_class': [2, 3],
    'suffix': ['bmp', 'jpg', 'png', 'ppm', 'jpeg'],
    'depth': [1, 2]}
))
class TestDirectoryParsingClassificationDataset(unittest.TestCase):

    n_img_per_class = 5

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        if self.depth == 1:
            _setup_depth_one_dummy_data(self.tmp_dir, self.n_class,
                                        self.n_img_per_class, self.size,
                                        self.color, self.suffix)
        elif self.depth == 2:
            _setup_depth_two_dummy_data(self.tmp_dir, self.n_class,
                                        self.n_img_per_class, self.size,
                                        self.color, self.suffix)

    def test_directory_parsing_classification_dataset(self):
        dataset = DirectoryParsingClassificationDataset(
            self.tmp_dir, color=self.color)

        self.assertEqual(len(dataset), self.n_img_per_class * self.n_class)

        assert_is_classification_dataset(
            dataset, self.n_class, color=self.color)

        img_names = [os.path.split(filename)[-1]
                     for filename in dataset.img_filenames]
        self.assertEqual(
            img_names[:self.n_img_per_class],
            ['img{}.{}'.format(i, self.suffix)
             for i in range(self.n_img_per_class)])

        label_names = directory_parsing_label_names(self.tmp_dir)
        self.assertEqual(
            label_names, ['class_{}'.format(i) for i in range(self.n_class)])


class TestNumericalSortDirectoryParsingClassificationDataset(
        unittest.TestCase):

    n_class = 11
    n_img_per_class = 1

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(self.n_class):
            class_dir = os.path.join(self.tmp_dir, '{}'.format(i))
            os.makedirs(class_dir)
            _save_img_file(os.path.join(class_dir, 'img_0.png'),
                           (48, 32), color=True)

    def test_numerical_sort(self):
        dataset = DirectoryParsingClassificationDataset(
            self.tmp_dir, numerical_sort=False)

        assert_is_classification_dataset(
            dataset, self.n_class)

        label_names = directory_parsing_label_names(
            self.tmp_dir, numerical_sort=True)
        self.assertEqual(
            label_names, ['{}'.format(i) for i in range(self.n_class)])


testing.run_module(__name__, __file__)
