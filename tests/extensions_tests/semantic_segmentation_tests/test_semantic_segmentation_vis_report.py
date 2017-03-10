import unittest

import mock
import numpy as np
import os.path as osp
import tempfile

from chainer import testing

from chainercv.extensions import SemanticSegmentationVisReport
from chainercv.testing import ConstantReturnModel
from chainercv.testing import DummyDataset


class TestSemanticSegmentationVisReport(unittest.TestCase):

    indices = [0, 1]

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.out_dir = tempfile.mkdtemp()
        self.trainer.out = self.out_dir
        self.trainer.updater.iteration = 0

        n_class = 2
        model = ConstantReturnModel(
            np.random.uniform(size=(1, 1, 10, 10)).astype(np.int32))
        dataset = DummyDataset(
            shapes=[(3, 10, 10), (1, 10, 10)],
            dtypes=[np.float32, np.int32])

        self.extension = SemanticSegmentationVisReport(
            self.indices, dataset, model, n_class=n_class,
            filename_base='semantic_seg')

    def test_call(self):
        self.extension(self.trainer)
        for idx in self.indices:
            file_name = osp.join(
                self.out_dir, 'semantic_seg_idx={}_iter=0.jpg'.format(idx))
            self.assertTrue(osp.exists(file_name))


testing.run_module(__name__, __file__)
