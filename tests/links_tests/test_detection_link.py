import numpy as np
import unittest

import chainer
from chainer import testing

from chainercv.links import DetectionLink


class TestDetectionLink(unittest.TestCase):

    def setUp(self):
        self.link = DetectionLink()
        self.img = np.random.randint(0, 255, (3, 32, 48))

    def test_detection_link(self):
        self.assertIsInstance(self.link, chainer.Link)

        with self.assertRaises(NotImplementedError):
            self.link.predict(self.img)


testing.run_module(__name__, __file__)
