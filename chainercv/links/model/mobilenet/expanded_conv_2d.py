import chainer
from chainer.functions import clipped_relu
from chainer.links import BatchNormalization

from chainercv.links.model.mobilenet import TFConv2DBNActiv
from chainercv.links.model.mobilenet.util import expand_input_by_factor


class ExpandedConv2D(chainer.Chain):
    """An expanded convolution 2d layer

    in --> expand conv (pointwise conv) --> depthwise conv --> project conv (pointwise conv) --> out

    Args:
        in_channels (int): The number of channels of the input array.
        out_channels (int): The number of channels of the output array.
        expand_pad (int, tuple of ints, 'SAME' or 'VALID'): Pad of expand conv filter application.
        depthwise_stride (int or tuple of ints): Stride of depthwise conv filter application.
        depthwise_ksize (int or tuple of ints): Kernel size of depthwise conv filter application.
        depthwise_pad (int, tuple of ints, 'SAME' or 'VALID'): Pad of depthwise conv filter application.
        project_pad (int, tuple of ints, 'SAME' or 'VALID'): Pad of project conv filter application.
        initialW (callable): Initial weight value used in
            the convolutional layers.
        bn_kwargs (dict): Keyword arguments passed to initialize
            :class:`chainer.links.BatchNormalization`.
    """
    def __init__(self,
                 out_channels,
                 in_channels,
                 expansion_size=expand_input_by_factor(6),
                 expand_pad='SAME',
                 depthwise_stride=1,
                 depthwise_ksize=3,
                 depthwise_pad='SAME',
                 project_pad='SAME',
                 initialW=None,
                 bn_kwargs={}):
        super(ExpandedConv2D, self).__init__()
        with self.init_scope():
            if callable(expansion_size):
                self.inner_size = expansion_size(num_inputs=in_channels)
            else:
                self.inner_size = expansion_size
            relu_six = lambda x: clipped_relu(x, 6.)
            if self.inner_size > in_channels:
                self.expand = TFConv2DBNActiv(
                    in_channels,
                    self.inner_size,
                    ksize=1,
                    pad=expand_pad,
                    nobias=True,
                    initialW=initialW,
                    bn_kwargs=bn_kwargs,
                    activ=relu_six)
                depthwise_in_channels = self.inner_size
            else:
                depthwise_in_channels = in_channels
            self.depthwise = TFConv2DBNActiv(
                depthwise_in_channels,
                self.inner_size,
                ksize=depthwise_ksize,
                stride=depthwise_stride,
                pad=depthwise_pad,
                nobias=True,
                initialW=initialW,
                groups=depthwise_in_channels,
                bn_kwargs=bn_kwargs,
                activ=relu_six)
            self.project = TFConv2DBNActiv(
                self.inner_size,
                out_channels,
                ksize=1,
                pad=project_pad,
                nobias=True,
                initialW=initialW,
                bn_kwargs=bn_kwargs,
                activ=None)

    def __call__(self, x):
        h = x
        if hasattr(self, "expand"):
            h = self.expand(x)
        h = self.depthwise(h)
        h = self.project(h)
        if h.shape == x.shape:
            h += x
        return h
