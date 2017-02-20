import unittest

import mock
import numpy as np
import os.path as osp
import tempfile

import chainer
from chainer import testing

from chainer_cv.extensions import MeasureKRetrieval
from chainer_cv.testing import DummyDataset


class TestMeasureKRetrieval(unittest.TestCase):

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.out_dir = tempfile.mkdtemp()
        self.trainer.out = self.out_dir
        self.trainer.observation = {}

        length = 10
        embed_dim = 5
        expected_shape = (length, embed_dim)
        features = np.random.uniform(size=expected_shape).astype(np.float32)
        features_file = osp.join(self.out_dir, 'embed.npy')
        np.save(features_file, features)

        dataset = DummyDataset(
            shapes=[(3, 10, 10), ()], dtypes=[np.float32, np.int32],
            length=length)
        iterator = chainer.iterators.SerialIterator(
            dataset, batch_size=2, repeat=False, shuffle=False)

        self.extension = MeasureKRetrieval(iterator, features_file, [2, 4])

    def test_call(self):
        self.extension(self.trainer)


testing.run_module(__name__, __file__)
