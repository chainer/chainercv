import unittest

import mock
import numpy as np
import os.path as osp
import tempfile

from chainer import testing

from chainercv.extensions import DetectionVisReport
from chainercv.utils import ConstantReturnModel
from chainercv.utils import DummyDataset


@testing.parameterize(
    {'shape': (3, 5)},
    {'shape': (0, 5)},
)
class TestDetectionVisReport(unittest.TestCase):

    indices = [0, 1]

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.out_dir = tempfile.mkdtemp()
        self.trainer.out = self.out_dir
        self.trainer.updater.iteration = 0

        model = ConstantReturnModel(np.random.uniform(size=(1,) + self.shape))
        dataset = DummyDataset(
            shapes=[(3, 10, 10), self.shape],
            dtypes=[np.float32, np.float32])

        self.extension = DetectionVisReport(
            self.indices, dataset, model,
            filename_base='detection')

    def test_call(self):
        self.extension(self.trainer)
        for idx in self.indices:
            file_name = osp.join(
                self.out_dir, 'detection_idx={}_iter=0.jpg'.format(idx))
            self.assertTrue(osp.exists(file_name))


testing.run_module(__name__, __file__)
