import numpy as np

import chainer
from chainer.functions import average_pooling_2d
from chainer.functions import clipped_relu
from chainer.functions import softmax
from chainer.functions import squeeze

from chainercv.links.model.mobilenet.expanded_conv_2d import ExpandedConv2D
from chainercv.links.model.mobilenet.tf_conv_2d_bn_activ import TFConv2DBNActiv
from chainercv.links.model.mobilenet.tf_convolution_2d import TFConvolution2D
from chainercv.links.model.mobilenet.util import _make_divisible
from chainercv.links.model.mobilenet.util import expand_input_by_factor
from chainercv.links.model.pickable_sequential_chain import \
    PickableSequentialChain
from chainercv import utils


"""
Implementation of Mobilenet V2, converting the weights from the pretrained
Tensorflow model from
https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
This MobileNetV2 implementation is based on @alexisVallet's one.
@okdshin modified it for ChainerCV.
"""


def _depth_multiplied_output_channels(base_out_channels,
                                      multiplier,
                                      divisable_by=8,
                                      min_depth=8):
    return _make_divisible(base_out_channels * multiplier, divisable_by,
                           min_depth)


_tf_mobilenetv2_mean = np.asarray(
    [128] * 3, dtype=np.float)[:, np.newaxis, np.newaxis]
_tf_mobilenetv2_scale = np.asarray(
    [1 / 128.0] * 3, dtype=np.float)[:, np.newaxis, np.newaxis]

# RGB order
_imagenet_mean = np.array(
    [123.68, 116.779, 103.939], dtype=np.float32)[:, np.newaxis, np.newaxis]
_imagenet_scale = np.array(
    [1.0]*3, dtype=np.float32)[:, np.newaxis, np.newaxis]


class MobileNetV2(PickableSequentialChain):
    """MobileNetV2 Network.

    This is a pickable sequential link.
    The network can choose output layers from set of all
    intermediate layers.
    The attribute :obj:`pick` is the names of the layers that are going
    to be picked by :meth:`__call__`.
    The attribute :obj:`layer_names` is the names of all layers
    that can be picked.

    Examples:

        >>> model = MobileNetV2()
        # By default, __call__ returns a probability score (after Softmax).
        >>> prob = model(imgs)
        >>> model.pick = 'expanded_conv_5'
        # This is layer expanded_conv_5.
        >>> expanded_conv_5 = model(imgs)
        >>> model.pick = ['expanded_conv_5', 'conv1']
        >>> # These are layers expanded_conv_5 and conv1 (before Pool).
        >>> expanded_conv_5, conv1 = model(imgs)

    .. seealso::
        :class:`chainercv.links.model.PickableSequentialChain`

    When :obj:`pretrained_model` is the path of a pre-trained chainer model
    serialized as a :obj:`.npz` file in the constructor, this chain model
    automatically initializes all the parameters with it.
    When a string in the prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`imagenet`: Loads weights trained with ImageNet. \
        When :obj:`arch=='tf'`, the weights distributed \
        at tensorflow/models
        `<https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet>`_ \ # NOQA
        are used.

    Args:
        n_class (int): The number of classes. If :obj:`None`,
            the default values are used.
            If a supported pretrained model is used,
            the number of classes used to train the pretrained model
            is used. Otherwise, the number of classes in ILSVRC 2012 dataset
            is used.
        pretrained_model (string): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        mean (numpy.ndarray): A mean value. If :obj:`None`,
            the default values are used.
            If a supported pretrained model is used,
            the mean value used to train the pretrained model is used.
            Otherwise, the mean value used by TF's implementation is used.
        initialW (callable): Initializer for the weights.
        initial_bias (callable): Initializer for the biases.

    """
    # Batch normalization replicating default tensorflow slim parameters
    # as used in the original tensorflow implementation.
    _bn_tf_default_params = {
        "decay": 0.999,
        "eps": 0.001,
        "dtype": chainer.config.dtype
    }

    _models = {
        'tf': {
            'param': {
                'n_class':
                1001,  # first element is background
                'mean': _tf_mobilenetv2_mean,
                'scale': _tf_mobilenetv2_scale,
            },
            'overwritable': (),
            'url': 'https://chainercv-models.preferred.jp/mobilenet_v2_depth_multiplier_1.0_imagenet_converted_2019_04_01.npz',  # NOQA
        }
    }

    def __init__(self,
                 n_class=None,
                 pretrained_model=None,
                 mean=None,
                 scale=None,
                 initialW=None,
                 initial_bias=None,
                 arch='tf',
                 depth_multiplier=1.,
                 bn_kwargs=_bn_tf_default_params,
                 thousand_categories_mode=False):
        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier must be greater than 0')

        param, path = utils.prepare_pretrained_model(
            {
                'n_class': n_class,
                'mean': mean,
                'scale': scale
            }, pretrained_model, self._models, {
                'n_class': 1000,
                'mean': _imagenet_mean,
                'scale': _imagenet_scale
            })
        self.mean = param['mean']
        self.scale = param['scale']
        self.n_class = param['n_class']

        super(MobileNetV2, self).__init__()

        def relu6(x):
            return clipped_relu(x, 6.)
        with self.init_scope():
            conv_out_channels = _depth_multiplied_output_channels(
                32, depth_multiplier)
            self.conv = TFConv2DBNActiv(
                in_channels=3,
                out_channels=conv_out_channels,
                stride=2,
                ksize=3,
                nobias=True,
                activ=relu6,
                initialW=initialW,
                bn_kwargs=bn_kwargs)
            expanded_out_channels = _depth_multiplied_output_channels(
                16, depth_multiplier)
            self.expanded_conv = ExpandedConv2D(
                expansion_size=expand_input_by_factor(1, divisible_by=1),
                in_channels=conv_out_channels,
                out_channels=expanded_out_channels,
                initialW=initialW,
                bn_kwargs=bn_kwargs)
            in_channels = expanded_out_channels
            out_channels_list = (24, ) * 2 + (32, ) * 3 + (64, ) * 4 + (
                96, ) * 3 + (160, ) * 3 + (320, )
            for i, out_channels in enumerate(out_channels_list):
                layer_id = i + 1
                if layer_id in (1, 3, 6, 13):
                    stride = 2
                else:
                    stride = 1
                multiplied_out_channels = _depth_multiplied_output_channels(
                    out_channels, depth_multiplier)
                setattr(self, "expanded_conv_{}".format(layer_id),
                        ExpandedConv2D(
                            in_channels=in_channels,
                            out_channels=multiplied_out_channels,
                            depthwise_stride=stride,
                            initialW=initialW,
                            bn_kwargs=bn_kwargs))
                in_channels = multiplied_out_channels
            if depth_multiplier < 1:
                conv1_out_channels = 1280
            else:
                conv1_out_channels = _depth_multiplied_output_channels(
                    1280, depth_multiplier)
            self.conv1 = TFConv2DBNActiv(
                in_channels=in_channels,
                out_channels=conv1_out_channels,
                ksize=1,
                nobias=True,
                initialW=initialW,
                activ=relu6,
                bn_kwargs=bn_kwargs)
            self.global_average_pool = \
                lambda x: average_pooling_2d(x, ksize=x.shape[2:4], stride=1)
            self.logits_conv = TFConvolution2D(
                in_channels=conv1_out_channels,
                out_channels=self.n_class,
                ksize=1,
                nobias=False,  # bias is needed
                initialW=initialW,
                initial_bias=initial_bias,
            )
            self.squeeze = lambda x: squeeze(x, axis=(2, 3))
            self.softmax = softmax

        if path:
            chainer.serializers.load_npz(path, self)

        if thousand_categories_mode and 1000 < n_class:
            self.logits_conv.W.data = np.delete(self.logits_conv.W.data, 0, 0)
            self.logits_conv.b.data = np.delete(self.logits_conv.b.data, 0)
