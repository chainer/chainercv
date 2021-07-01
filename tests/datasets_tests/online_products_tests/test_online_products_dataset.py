import unittest

import numpy as np

from chainer import testing
from chainer.testing import attr

from chainercv.datasets import online_products_super_label_names
from chainercv.datasets import OnlineProductsDataset
from chainercv.utils import assert_is_label_dataset


@testing.parameterize(
    {'split': 'train'},
    {'split': 'test'}
)
class TestOnlineProductsDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = OnlineProductsDataset(split=self.split)

    @attr.slow
    def test_online_products_dataset(self):
        assert_is_label_dataset(
            self.dataset, 22634, n_example=10)

        for _ in range(10):
            i = np.random.randint(0, len(self.dataset))
            _, _, super_label = self.dataset[i]

            assert isinstance(super_label, np.int32), \
                'label must be a numpy.int32.'
            assert super_label.ndim == 0, 'The ndim of label must be 0'
            assert (super_label >= 0 and
                    super_label < len(online_products_super_label_names)), \
                'The value of label must be in [0, n_class - 1].'


testing.run_module(__name__, __file__)
