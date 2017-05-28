import numpy as np
import unittest

import chainer
from chainer import testing
from chainer.testing import attr

from chainercv.links import SegNetBasic


@testing.parameterize(
    {'train': False},
    {'train': True}
)
@attr.slow
class TestSegNetBasic(unittest.TestCase):

    def setUp(self):
        self.n_class = 10
        self.link = SegNetBasic(n_class=self.n_class)

    def check_call(self):
        xp = self.link.xp
        x = chainer.Variable(xp.random.uniform(
            low=-1, high=1, size=(2, 3, 128, 160)).astype(np.float32))
        y = self.link(x)

        self.assertIsInstance(y, chainer.Variable)
        self.assertIsInstance(y.data, xp.ndarray)
        self.assertEqual(y.shape, (2, self.n_class, 128, 160))

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()

    def check_predict(self):
        hs = np.random.randint(128, 160, size=(2,))
        ws = np.random.randint(128, 160, size=(2,))
        imgs = [
            np.random.uniform(size=(3, hs[0], ws[0])).astype(np.float32),
            np.random.uniform(size=(3, hs[1], ws[1])).astype(np.float32),
        ]

        labels = self.link.predict(imgs)

        self.assertEqual(len(labels), 2)
        for i in range(2):
            self.assertIsInstance(labels[i], np.ndarray)
            self.assertEqual(labels[i].shape, (hs[i], ws[i]))
            self.assertEqual(labels[i].dtype, np.int64)

    def test_predict_cpu(self):
        self.check_predict()

    @attr.gpu
    def test_predict_gpu(self):
        self.link.to_gpu()
        self.check_predict()


testing.run_module(__name__, __file__)
