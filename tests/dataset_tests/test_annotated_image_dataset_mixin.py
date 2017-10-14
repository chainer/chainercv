import unittest

import numpy as np

from chainer import testing
from chainercv.dataset import AnnotatedImageDatasetMixin


class SingleAnnotationDataset(AnnotatedImageDatasetMixin):
    def __len__(self):
        return 10

    def get_image(self, i):
        return np.random.uniform(0, 256, size=(3, 32, 48)).astype(np.float32)

    def get_annotation(self, i):
        return np.random.randint(0, 9, dtype=np.int32)


class MultipleAnnotationDataset(AnnotatedImageDatasetMixin):
    def __len__(self):
        return 10

    def get_image(self, i):
        return np.random.uniform(0, 256, (3, 32, 48)).astype(np.float32)

    def get_annotation(self, i):
        anno0 = np.random.uniform(0, 32, size=(20, 4), dtype=np.float32)
        anno1 = np.random.randint(0, 9, size=20, dtype=np.int32)
        return anno0, anno1


@testing.parameterize(
    {'dataset': SingleAnnotationDataset, 'n_anno': 1},
    {'dataset': MultipleAnnotationDataset, 'n_anno': 2},
)
class TestAnnotatedImageDatasetMixin(unittest.TestCase):

    def setup(self):
        self.dataset = self.dataset()

    def test_base_dataset_len(self):
        self.assertEqual(len(self.dataset), 10)

    def test_base_dataset_index(self):
        for i in range(self.dataset):
            example = self.dataset[i]
            self.assertEqual(len(example), 1 + self.n_anno)

    def test_base_dataset_slice(self):
        for example in self.dataset[:]:
            self.assertEqual(len(example), 1 + self.n_anno)

    def test_annotation_dataset_len(self):
        self.assertEqual(len(self.dataset.annotations), 10)

    def test_annotation_dataset_index(self):
        for i in range(self.dataset.annotations):
            example = self.dataset.annotations[i]
            self.assertEqual(len(example), self.n_anno)

    def test_annotation_dataset_slice(self):
        for example in self.dataset.annotations[:]:
            self.assertEqual(len(example), self.n_anno)


testing.run_module(__name__, __file__)
