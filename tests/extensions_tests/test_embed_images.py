import unittest

import mock
import numpy as np
import os.path as osp
import tempfile

import chainer
from chainer import testing

from chainer_cv.extensions import EmbedImages
from chainer_cv.testing import ConstantReturnModel
from chainer_cv.testing import DummyDataset


class TestEmbedImages(unittest.TestCase):

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.out_dir = tempfile.mkdtemp()
        self.trainer.out = self.out_dir
        self.filename = 'embed.npy'

        length = 100
        embed_dim = 50
        self.expected_shape = (length, embed_dim)

        model = ConstantReturnModel(
            np.random.uniform(size=(2, embed_dim)).astype(np.float32))
        dataset = DummyDataset(
            shapes=[(3, 10, 10), ()],
            dtypes=[np.float32, np.int32], length=length)
        iterator = chainer.iterators.SerialIterator(
            dataset, batch_size=2, repeat=False, shuffle=False)

        self.extension = EmbedImages(iterator, model, filename=self.filename)

    def test_call(self):
        self.extension(self.trainer)
        out_file = osp.join(self.out_dir, self.filename)
        self.assertTrue(osp.exists(out_file))

        embedded_feats = np.load(out_file)
        self.assertEqual(embedded_feats.shape, self.expected_shape)


testing.run_module(__name__, __file__)
