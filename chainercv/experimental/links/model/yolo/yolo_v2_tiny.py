import chainer

from chainercv.links import Conv2DBNActiv

from chainercv.links.model.yolo.yolo_v2 import _leaky_relu
from chainercv.links.model.yolo.yolo_v2 import _maxpool
from chainercv.links.model.yolo import YOLOv2Base


class DarknetExtractor(chainer.ChainList):
    """A Darknet based feature extractor for YOLOv2Tiny.


    This is a feature extractor for
    :class:`~chainercv.links.model.yolo.YOLOv2Tiny`
    """

    insize = 416
    grid = 13

    def __init__(self):
        super(DarknetExtractor, self).__init__()

        # Darknet
        for k in range(7):
            self.append(Conv2DBNActiv(16 << k, 3, pad=1, activ=_leaky_relu))

        # additional link
        self.append(Conv2DBNActiv(1024, 3, pad=1, activ=_leaky_relu))

    def forward(self, x):
        """Compute a feature map from a batch of images.

        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized to :math:`416\\times 416`.

        Returns:
            Variable:
        """

        h = x
        for i, link in enumerate(self):
            h = link(h)
            if i < 5:
                h = _maxpool(h, 2)
            elif i == 5:
                h = _maxpool(h, 2, stride=1)
        return h


class YOLOv2Tiny(YOLOv2Base):
    """YOLOv2 tiny.

    This is a model of YOLOv2 tiny a.k.a. Tiny YOLO.
    This model uses :class:`~chainercv.links.model.yolo.DarknetExtractor` as
    its feature extractor.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (string): The weight file to be loaded.
            This can take :obj:`'voc0712'`, `filepath` or :obj:`None`.
            The default value is :obj:`None`.

            * :obj:`'voc0712'`: Load weights trained on trainval split of \
                PASCAL VOC 2007 and 2012. \
                The weight file is downloaded and cached automatically. \
                :obj:`n_fg_class` must be :obj:`20` or :obj:`None`. \
                These weights were converted from the darknet model \
                provided by `the original implementation \
                <https://pjreddie.com/darknet/yolov2/>`_. \
                The conversion code is \
                `chainercv/examples/yolo/darknet2npz.py`.
            * `filepath`: A path of npz file. In this case, :obj:`n_fg_class` \
                must be specified properly.
            * :obj:`None`: Do not load weights.

    """

    _extractor = DarknetExtractor

    _models = {
        'voc0712': {
            'param': {'n_fg_class': 20},
            'url': 'https://chainercv-models.preferred.jp/'
            'yolo_v2_tiny_voc0712_converted_2018_10_19.npz',
            'cv2': True
        },
    }

    _anchors = (
        (1.19, 1.08),
        (4.41, 3.42),
        (11.38, 6.63),
        (5.11, 9.42),
        (10.52, 16.62))
