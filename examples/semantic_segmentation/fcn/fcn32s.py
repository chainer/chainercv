import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links.model.vision.vgg import VGG16Layers

from chainer_cv.evaluations.eval_semantic_segmentation import \
    label_accuracy_score


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


class FCN32s(chainer.Chain):

    """Full Convolutional Network 32s

    Fully connected layers of VGG networks is used for their equivalent
    convolutional layers.
    Equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.

    TODO:
        In the original fully convoltional neural networks implementation,
        learning rate of bias term and W term differ. This is not implemented.
        ``https://github.com/shelhamer/fcn.berkeleyvision.org/
        blob/master/voc-fcn32s/net.py``.

        Also, learning rate of the last layer is set to 0.

    """

    def __init__(self, n_class=21):
        self.n_class = n_class
        deconv_filter = upsample_filt(64)
        deconv_filter = np.broadcast_to(
            deconv_filter, (self.n_class, self.n_class, 64, 64))
        super(self.__class__, self).__init__(
            vgg=VGG16Layers(),
            fc6=L.Convolution2D(512, 4096, 7, stride=1, pad=0),
            fc7=L.Convolution2D(4096, 4096, 1, stride=1, pad=0),

            score_fr=L.Convolution2D(4096, self.n_class, 1, stride=1, pad=0),

            upscore=L.Deconvolution2D(self.n_class, self.n_class, 64,
                                      stride=32, pad=0, initialW=deconv_filter,
                                      nobias=True),
        )
        # copy fully connected layers' weights to equivalent conv layers
        self.fc6.W.data[:] = self.vgg.fc6.W.data.reshape(self.fc6.W.shape)
        self.fc6.b.data[:] = self.vgg.fc6.b.data.reshape(self.fc6.b.shape)
        self.fc7.W.data[:] = self.vgg.fc7.W.data.reshape(self.fc7.W.shape)
        self.fc7.b.data[:] = self.vgg.fc7.b.data.reshape(self.fc7.b.shape)

        self.vgg.conv1_1.pad = 100

        self.train = False

    def __call__(self, x, t=None):
        pool5 = self.vgg(x, layers=['pool5'])['pool5']

        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=.5, train=self.train)

        h = F.relu(self.fc7(h))
        h = F.dropout(h, ratio=.5, train=self.train)

        h = self.score_fr(h)
        h = self.upscore(h)
        self.score = h[:, :, 19:19 + x.data.shape[2], 19:19 + x.data.shape[3]]

        if t is None:
            return self.score

        # testing with t or training
        self.loss = F.softmax_cross_entropy(
            self.score, t[:, 0, :, :], normalize=False)

        xp = chainer.cuda.get_array_module(self.loss.data)
        if xp.isnan(self.loss.data):
            raise ValueError('loss value is nan')

        # report the loss and accuracy
        labels = chainer.cuda.to_cpu(t.data)
        label_preds = chainer.cuda.to_cpu(self.score.data).argmax(axis=1)
        results = []
        for i in xrange(x.shape[0]):
            acc, acc_cls, iu, fwavacc = label_accuracy_score(
                labels[i][0], label_preds[i], self.n_class)
            results.append((acc, acc_cls, iu, fwavacc))
        results = np.array(results).mean(axis=0)
        chainer.reporter.report({
            'loss': self.loss,
            'accuracy': results[0],
            'accuracy_cls': results[1],
            'iu': results[2],
            'fwavacc': results[3],
        }, self)

        return self.loss

    def extract(self, x, t):
        self.__call__(x, t)
        return self.score
