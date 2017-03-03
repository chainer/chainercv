from collections import OrderedDict
import os

from chainer.dataset import download
from chainer.functions.activation.relu import relu
from chainer.functions.noise.dropout import dropout
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from chainer.functions.normalization.local_response_normalization\
    import local_response_normalization
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.inception import Inception
from chainer.links.connection.linear import Linear
from chainer.serializers import npz


class GoogLeNet(link.Chain):
    """A pre-trained CNN model of GoogLeNet [1].

    During initialization, this chain model automatically downloads
    the pre-trained caffemodel, convert to another chainer model,
    stores it on your local directory, and initializes all the parameters
    with it. This model would be useful when you want to extract a semantic
    feature vector from a given image, or fine-tune the model
    on a different dataset.
    Note that this pre-trained model is released under Creative Commons
    Attribution License.

    If you want to manually convert the pre-trained caffemodel to a chainer
    model that can be specified in the constructor,
    please use ``convert_caffemodel_to_npz`` classmethod instead.

    .. [1] K. Simonyan and A. Zisserman, `Going Deeper with Convolutions
        <https://arxiv.org/abs/1409.4842>`_

    Args:
        pretrained_model (str): the destination of the pre-trained
            chainer model serialized as a ``.npz`` file.
            If this argument is specified as ``auto``,
            it automatically loads and converts the caffemodel from
            ``$CHAINER_DATASET_ROOT/yuyu2172/chainer-cv/models/
            bvlc_googlenet.caffemodel``,
            where ``$CHAINER_DATASET_ROOT`` is set as
            ``$HOME/.chainer/dataset`` unless you specify another value
            as an environment variable. Note that in this case the converted
            chainer model is stored on the same directory and automatically
            used from the second time.
            If the argument is specified as ``None``, all the parameters
            are not initialized by the pre-trained model.

    Attributes:
        available_layers (list of str): The list of available layer names
            used by ``__call__`` and ``extract`` methods.
    """

    def __init__(self, pretrained_model='auto'):
        super(GoogLeNet, self).__init__(
            conv1=Convolution2D(3,  64, 7, stride=2, pad=3),
            conv2_reduce=Convolution2D(64,  64, 1),
            conv2=Convolution2D(64, 192, 3, stride=1, pad=1),
            inc3a=Inception(192,  64,  96, 128, 16,  32,  32),
            inc3b=Inception(256, 128, 128, 192, 32,  96,  64),
            inc4a=Inception(480, 192,  96, 208, 16,  48,  64),
            inc4b=Inception(512, 160, 112, 224, 24,  64,  64),
            inc4c=Inception(512, 128, 128, 256, 24,  64,  64),
            inc4d=Inception(512, 112, 144, 288, 32,  64,  64),
            inc4e=Inception(528, 256, 160, 320, 32, 128, 128),
            inc5a=Inception(832, 256, 160, 320, 32, 128, 128),
            inc5b=Inception(832, 384, 192, 384, 48, 128, 128),
            loss3_fc=Linear(1024, 1000),

            loss1_conv=Convolution2D(512, 128, 1),
            loss1_fc1=Linear(4 * 4 * 128, 1024),
            loss1_fc2=Linear(1024, 1000),

            loss2_conv=Convolution2D(528, 128, 1),
            loss2_fc1=Linear(4 * 4 * 128, 1024),
            loss2_fc2=Linear(1024, 1000),
        )
        if pretrained_model == 'auto':
            _retrieve(
                'bvlc_googlenet.npz',
                'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
                self)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)
        self.functions = OrderedDict([
            ('conv1', [self.conv1, relu]),
            ('pool1', [lambda x: max_pooling_2d(x, ksize=3, stride=2),
                       lambda x: local_response_normalization(x, n=5)]),
            ('conv2_reduce', [self.conv2_reduce, relu]),
            ('conv2', [self.conv2, relu]),
            ('pool2', [lambda x: local_response_normalization(x, n=5),
                       lambda x: max_pooling_2d(x, ksize=3, stride=2)]),
            ('inc3a', [self.inc3a]),
            ('inc3b', [self.inc3b]),
            ('pool3', [lambda x: max_pooling_2d(x, ksize=3, stride=2)]),
            ('inc4a', [self.inc4a]),

            ('inc4b', [self.inc4b]),
            ('inc4c', [self.inc4c]),
            ('inc4d', [self.inc4d]),
            ('inc4e', [self.inc4e]),
            ('pool4', [lambda x: max_pooling_2d(x, ksize=3, stride=2)]),
            ('inc5a', [self.inc5a]),
            ('inc5b', [self.inc5b]),
            ('pool6', [lambda x: average_pooling_2d(x, ksize=7, stride=1)]),
            ('prob', [lambda x: dropout(x, ratio=0.4),
                      self.loss3_fc])
        ])

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.

        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_model=None)
        _transfer_googlenet(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def __call__(self, x, layers=['prob'], test=True):
        """Computes all the feature maps specified by ``layers``.

        Args:
            x (~chainer.Variable): Input variable.
            layers (list of str): The list of layer names you want to extract.
            test (bool): If ``True``, dropout runs in test mode.

        Returns:
            Dictionary of ~chainer.Variable: A directory in which
            the key contains the layer name and the value contains
            the corresponding feature map variable.

        """

        h = x
        activations = {}
        target_layers = set(layers)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                if func is dropout:
                    h = func(h, train=not test)
                else:
                    h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)
        return activations


def _transfer_Wb(src, dst):
    # transfer weights for Convolution and Linear layers
    dst.W.data[:] = src.W.data
    dst.b.data[:] = src.b.data


def _transfer_inception(src, dst, inception_name):
    _transfer_Wb(src[inception_name + '/' + '1x1'], dst.conv1)
    _transfer_Wb(src[inception_name + '/' + '3x3'], dst.conv3)
    _transfer_Wb(src[inception_name + '/' + '5x5'], dst.conv5)
    _transfer_Wb(src[inception_name + '/' + '3x3_reduce'], dst.proj3)
    _transfer_Wb(src[inception_name + '/' + '5x5_reduce'], dst.proj5)
    _transfer_Wb(src[inception_name + '/' + 'pool_proj'], dst.projp)


def _transfer_googlenet(src, dst):
    _transfer_Wb(src['conv1/7x7_s2'], dst.conv1)
    _transfer_Wb(src['conv2/3x3_reduce'], dst.conv2_reduce)
    _transfer_Wb(src['conv2/3x3'], dst.conv2)

    _transfer_inception(src, dst.inc3a, 'inception_3a')
    _transfer_inception(src, dst.inc3b, 'inception_3b')
    _transfer_inception(src, dst.inc4a, 'inception_4a')
    _transfer_inception(src, dst.inc4b, 'inception_4b')
    _transfer_inception(src, dst.inc4c, 'inception_4c')
    _transfer_inception(src, dst.inc4d, 'inception_4d')
    _transfer_inception(src, dst.inc4e, 'inception_4e')
    _transfer_inception(src, dst.inc5a, 'inception_5a')
    _transfer_inception(src, dst.inc5b, 'inception_5b')

    _transfer_Wb(src['loss3/classifier'], dst.loss3_fc)
    _transfer_Wb(src['loss1/conv'], dst.loss1_conv)
    _transfer_Wb(src['loss1/fc'], dst.loss1_fc1)
    _transfer_Wb(src['loss1/classifier'], dst.loss1_fc2)
    _transfer_Wb(src['loss2/conv'], dst.loss2_conv)
    _transfer_Wb(src['loss2/fc'], dst.loss2_fc1)
    _transfer_Wb(src['loss2/classifier'], dst.loss2_fc2)


def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    print('Now loading caffemodel (usually it may take few minutes)')
    GoogLeNet.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name, url, model):
    root = download.get_dataset_directory('yuyu2172/chainer-cv/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))
