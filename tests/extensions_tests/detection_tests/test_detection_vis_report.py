import mock
import numpy as np
import os.path as osp
import six
import tempfile
import unittest


from chainer.datasets import TupleDataset
from chainer import testing

from chainercv.extensions import DetectionVisReport
from chainercv.links import DetectionLink

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


class _RandomDetectionStubLink(DetectionLink):

    def predict(self, img):
        n_bbox = np.random.randint(0, 10)
        bbox = np.random.uniform(size=(n_bbox, 4))
        label = np.random.randint(0, 19, size=n_bbox)
        score = np.random.uniform(0, 1, size=n_bbox)
        return bbox, label, score


class TestDetectionVisReport(unittest.TestCase):

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.trainer.out = tempfile.mkdtemp()
        self.trainer.updater.iteration = 0

        self.link = _RandomDetectionStubLink()
        self.dataset = TupleDataset(
            np.random.uniform(size=(10, 3, 32, 48)),
            np.random.uniform(size=(10, 5, 4)),
            np.random.randint(0, 19, size=(10, 5)))

    def test_available(self):
        self.extension = DetectionVisReport(self.dataset, self.link)
        self.assertEqual(self.extension.available(), optional_modules)

    def test_basic(self):
        self.extension = DetectionVisReport(self.dataset, self.link)
        self.extension(self.trainer)

        if not optional_modules:
            return

        for idx in six.moves.range(len(self.dataset)):
            out_file = osp.join(
                self.trainer.out, 'detection_idx={:d}_iter=0.jpg'.format(idx))
            self.assertTrue(osp.exists(out_file))

    def test_with_filename(self):
        self.extension = DetectionVisReport(
            self.dataset, self.link,
            filename='result_iter_{iteration}_no_{index}.png')
        self.extension(self.trainer)

        if not optional_modules:
            return

        for idx in six.moves.range(len(self.dataset)):
            out_file = osp.join(
                self.trainer.out, 'result_iter_0_no_{:d}.png'.format(idx))
            self.assertTrue(osp.exists(out_file))


testing.run_module(__name__, __file__)
