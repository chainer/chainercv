import mock
import numpy as np
import os
import six
import tempfile
import unittest

import chainer
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from chainer import testing
from chainer.testing import attr

from chainercv.extensions import DetectionVisReport

try:
    import matplotlib  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


class _RandomDetectionStubLink(chainer.Link):

    def predict(self, imgs):
        bboxes = list()
        labels = list()
        scores = list()

        for _ in imgs:
            n_bbox = np.random.randint(0, 10)
            bboxes.append(self.xp.array(np.random.uniform(size=(n_bbox, 4))))
            labels.append(self.xp.array(np.random.randint(0, 19, size=n_bbox)))
            scores.append(self.xp.array(np.random.uniform(0, 1, size=n_bbox)))

        return bboxes, labels, scores


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
        self.iterator = SerialIterator(
            self.dataset, 10, repeat=False, shuffle=False)

    def test_available(self):
        self.extension = DetectionVisReport(self.dataset, self.link)
        self.assertEqual(self.extension.available(), optional_modules)

    def _check(self, filename='detection_iter=0_idx={:d}.jpg'):
        self.extension(self.trainer)

        if not optional_modules:
            return

        for idx in six.moves.range(len(self.dataset)):
            out_file = os.path.join(
                self.trainer.out, filename.format(idx))
            self.assertTrue(os.path.exists(out_file))

    def test_cpu(self):
        self.extension = DetectionVisReport(self.iterator, self.link)
        self._check()

    @attr.gpu
    def test_gpu(self):
        self.link.to_gpu()
        self.extension = DetectionVisReport(self.iterator, self.link)
        self._check()

    def test_with_filename(self):
        self.extension = DetectionVisReport(
            self.iterator, self.link,
            filename='result_no_{index}_iter_{iteration}.png')
        self._check('result_no_{:d}_iter_0.png')


testing.run_module(__name__, __file__)
