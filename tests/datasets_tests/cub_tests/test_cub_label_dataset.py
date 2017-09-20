import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import CUBLabelDataset
from chainercv.datasets import cub_label_names
from chainercv.utils import assert_is_classification_dataset
from chainercv.utils import assert_is_bbox


@testing.parameterize(
    {'return_bb': True},
    {'return_bb': False}
)
class TestCUBLabelDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = CUBLabelDataset(return_bb=self.return_bb)

    @attr.slow
    def test_cub_label_dataset(self):
        assert_is_classification_dataset(
            self.dataset, len(cub_label_names), n_example=10)
        if self.return_bb:
            idx = np.random.choice(np.arange(10))
            _, _, bb = self.dataset[idx]
            assert_is_bbox(bb[None])


testing.run_module(__name__, __file__)
