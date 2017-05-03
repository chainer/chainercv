import unittest

import mock
import numpy as np
import os.path as osp
import tempfile

from chainer.datasets import TupleDataset
from chainer import testing

from chainercv.extensions import DetectionVisReport
from chainercv.utils import ConstantStubLink

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


@testing.parameterize(
    {'bbox_shape': (3, 4), 'label_shape': (3,)},
    {'bbox_shape': (0, 4), 'label_shape': (0,)},
)
class TestDetectionVisReport(unittest.TestCase):

    indices = [0, 1]

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.out_dir = tempfile.mkdtemp()
        self.trainer.out = self.out_dir
        self.trainer.updater.iteration = 0

        model = ConstantStubLink((
            np.random.uniform(size=(1,) + self.bbox_shape).astype(np.float32),
            np.random.uniform(size=(1,) + self.label_shape).astype(np.int32)))
        dataset = TupleDataset(
            np.random.uniform(size=(100, 3, 10, 10)).astype(np.float32),
            np.random.uniform(
                size=(100,) + self.bbox_shape).astype(np.float32),
            np.random.uniform(
                size=(100,) + (self.label_shape)).astype(np.int32))

        self.extension = DetectionVisReport(
            self.indices, dataset, model,
            filename_base='detection')

    def test_call(self):
        self.extension(self.trainer)
        if optional_modules:
            for idx in self.indices:
                file_name = osp.join(
                    self.out_dir, 'detection_idx={}_iter=0.jpg'.format(idx))
                self.assertTrue(osp.exists(file_name))


testing.run_module(__name__, __file__)
