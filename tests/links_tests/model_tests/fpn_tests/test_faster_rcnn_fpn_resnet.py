import unittest

from chainer import testing
from chainer.testing import attr

from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv.utils import assert_is_detection_link


@testing.parameterize(*testing.product({
    'model': [FasterRCNNFPNResNet50, FasterRCNNFPNResNet101],
    'n_fg_class': [1, 5, 20],
}))
class TestFasterRCNNFPNResNet(unittest.TestCase):

    def setUp(self):
        self.link = self.model(n_fg_class=self.n_fg_class)

    @attr.slow
    def test_call_cpu(self):
        assert_is_detection_link(self.link, self.n_fg_class)

    @attr.gpu
    @attr.slow
    def test_call_gpu(self):
        self.link.to_gpu()
        assert_is_detection_link(self.link, self.n_fg_class)


@testing.parameterize(*testing.product({
    'model': [FasterRCNNFPNResNet50, FasterRCNNFPNResNet101],
    'n_fg_class': [None, 10, 80],
    'pretrained_model': ['coco', 'imagenet'],
}))
class TestFasterRCNNFPNResNetPretrained(unittest.TestCase):

    @attr.slow
    def test_pretrained(self):
        kwargs = {
            'n_fg_class': self.n_fg_class,
            'pretrained_model': self.pretrained_model,
        }

        if self.pretrained_model == 'coco':
            valid = self.n_fg_class in {None, 80}
        elif self.pretrained_model == 'imagenet':
            valid = self.n_fg_class is not None

        if valid:
            self.model(**kwargs)
        else:
            with self.assertRaises(ValueError):
                self.model(**kwargs)


testing.run_module(__name__, __file__)
