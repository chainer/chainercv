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
from chainercv.utils import generate_random_bbox

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
            bboxes.append(generate_random_bbox(
                n_bbox, (48, 32), 4, 12))
            labels.append(np.random.randint(0, 19, size=n_bbox))
            scores.append(np.random.uniform(0, 1, size=n_bbox))

        return bboxes, labels, scores


@testing.parameterize(
    {
        'filename': None,
        'filename_func': lambda iter_, idx:
        'detection_iter={:d}_idx={:d}.jpg'.format(iter_, idx)},
    {
        'filename': 'result_no_{index}_iter_{iteration}.png',
        'filename_func': lambda iter_, idx:
        'result_no_{:d}_iter_{:d}.png'.format(idx, iter_)},
    {
        'filename': 'detection_iter={iteration}.jpg',
        'filename_func': lambda iter_, _:
        'detection_iter={:d}.jpg'.format(iter_)},
    {
        'filename': 'detection_idx={index}.jpg',
        'filename_func': lambda _, idx:
        'detection_idx={:d}.jpg'.format(idx)},
)
class TestDetectionVisReport(unittest.TestCase):

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.trainer.out = tempfile.mkdtemp()

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

    def _check(self):
        if self.filename is None:
            extension = DetectionVisReport(self.iterator, self.link)
        else:
            extension = DetectionVisReport(
                self.iterator, self.link, filename=self.filename)

        if not optional_modules:
            return

        for iter_ in range(3):
            self.trainer.updater.iteration = iter_
            extension(self.trainer)

            for idx in six.moves.range(len(self.dataset)):
                out_file = os.path.join(
                    self.trainer.out, self.filename_func(iter_, idx))
                self.assertTrue(os.path.exists(out_file))

    def test_cpu(self):
        self._check()

    @attr.gpu
    def test_gpu(self):
        self.link.to_gpu()
        self._check()


testing.run_module(__name__, __file__)
