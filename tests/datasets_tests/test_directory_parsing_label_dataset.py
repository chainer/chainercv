import numpy as np
import os
import shutil
import tempfile
import unittest

from chainer import testing

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.utils import assert_is_label_dataset
from chainercv.utils import write_image


def _save_img_file(path, size, color):
    if color:
        img = np.random.randint(
            0, 255, size=(3,) + size, dtype=np.uint8)
    else:
        img = np.random.randint(
            0, 255, size=(1,) + size, dtype=np.uint8)
    write_image(img, path)


def _setup_depth_one_dummy_data(tmp_dir, n_class, n_img_per_class,
                                size, color, suffix):
    for i in range(n_class):
        class_dir = os.path.join(tmp_dir, 'class_{}'.format(i))
        os.makedirs(class_dir)
        for j in range(n_img_per_class):
            path = os.path.join(class_dir, 'img{}.{}'.format(j, suffix))
            _save_img_file(path, size, color)
        open(os.path.join(class_dir, 'dummy_file.XXX'), 'a').close()


def _setup_depth_two_dummy_data(tmp_dir, n_class, n_img_per_class,
                                n_sub_directory, size, color, suffix):
    for i in range(n_class):
        class_dir = os.path.join(tmp_dir, 'class_{}'.format(i))
        os.makedirs(class_dir)
        for j in range(n_sub_directory):
            nested_dir = os.path.join(class_dir, 'nested_{}'.format(j))
            os.makedirs(nested_dir)
            for k in range(n_img_per_class):
                path = os.path.join(
                    nested_dir, 'img{}.{}'.format(k, suffix))
                _save_img_file(path, size, color)
            open(os.path.join(nested_dir, 'dummy_file.XXX'), 'a').close()


@testing.parameterize(*testing.product({
    'size': [(48, 32)],
    'color': [True, False],
    'n_class': [2, 3],
    'suffix': ['bmp', 'jpg', 'png', 'ppm', 'jpeg'],
    'depth': [1, 2]}
))
class TestDirectoryParsingLabelDataset(unittest.TestCase):

    n_img_per_class = 5
    n_sub_directory = 6

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        if self.depth == 1:
            _setup_depth_one_dummy_data(self.tmp_dir, self.n_class,
                                        self.n_img_per_class, self.size,
                                        self.color, self.suffix)
        elif self.depth == 2:
            _setup_depth_two_dummy_data(self.tmp_dir, self.n_class,
                                        self.n_img_per_class,
                                        self.n_sub_directory, self.size,
                                        self.color, self.suffix)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_directory_parsing_label_dataset(self):
        dataset = DirectoryParsingLabelDataset(
            self.tmp_dir, color=self.color)

        if self.depth == 1:
            expected_legnth = self.n_img_per_class * self.n_class
        elif self.depth == 2:
            expected_legnth =\
                self.n_img_per_class * self.n_sub_directory * self.n_class
        self.assertEqual(len(dataset), expected_legnth)

        assert_is_label_dataset(dataset, self.n_class, color=self.color)

        label_names = directory_parsing_label_names(self.tmp_dir)
        self.assertEqual(
            label_names, ['class_{}'.format(i) for i in range(self.n_class)])

        if self.depth == 1:
            self.assertEqual(
                dataset.img_paths,
                ['{}/class_{}/img{}.{}'.format(self.tmp_dir, i, j, self.suffix)
                 for i in range(self.n_class)
                 for j in range(self.n_img_per_class)])
        elif self.depth == 2:
            self.assertEqual(
                dataset.img_paths,
                ['{}/class_{}/nested_{}/img{}.{}'.format(
                    self.tmp_dir, i, j, k, self.suffix)
                 for i in range(self.n_class)
                 for j in range(self.n_sub_directory)
                 for k in range(self.n_img_per_class)])


class TestNumericalSortDirectoryParsingLabelDataset(
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

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_numerical_sort(self):
        dataset = DirectoryParsingLabelDataset(
            self.tmp_dir, numerical_sort=True)

        assert_is_label_dataset(dataset, self.n_class)

        label_names = directory_parsing_label_names(
            self.tmp_dir, numerical_sort=True)
        self.assertEqual(
            label_names, ['{}'.format(i) for i in range(self.n_class)])


testing.run_module(__name__, __file__)
