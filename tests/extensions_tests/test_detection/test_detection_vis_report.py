import unittest

import mock
import numpy as np
import os.path as osp
import tempfile

from chainer import testing

from chainer_cv.testing import DummyDatasetGetRawData
from chainer_cv.testing import ConstantReturnModel
from chainer_cv.extensions import DetectionVisReport


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
        dataset = DummyDatasetGetRawData(
            shapes=[(3, 10, 10), self.shape],
            get_raw_data_shapes=[(10, 10, 3), self.shape],
            dtypes=[np.float32, np.float32],
            get_raw_data_dtypes=[np.uint8, np.float32])

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
