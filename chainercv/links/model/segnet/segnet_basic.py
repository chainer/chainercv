import chainer
import chainer.functions as F
import chainer.links as L


class SegNetBasic(chainer.Chain):

    """SegNet Basic for semantic segmentation.

    This is a SegNet [#]_ model for semantic segmenation. This is based on
    SegNetBasic model that is found here_.

    .. [#] Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A \
    Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." \
    PAMI, 2017 

    .. _here: http://github.com/alexgkendall/SegNet-Tutorial

    Args:
        out_ch (int): The number of channels for the final convolutional layer.
            SegNetBasic basically takes the number of target classes as this
            argment.
    """

    def __init__(self, out_ch):
        w = chainer.initializers.HeNormal()
        super(SegNetBasic, self).__init__(
            conv1=L.Convolution2D(None, 64, 7, 1, 3, nobias=True, initialW=w),
            conv1_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv2=L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=w),
            conv2_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv3=L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=w),
            conv3_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv4=L.Convolution2D(64, 64, 7, 1, 3, nobias=True, initialW=w),
            conv4_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv_decode4=L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=w),
            conv_decode4_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv_decode3=L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=w),
            conv_decode3_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv_decode2=L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=w),
            conv_decode2_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv_decode1=L.Convolution2D(
                64, 64, 7, 1, 3, nobias=True, initialW=w),
            conv_decode1_bn=L.BatchNormalization(64, initial_beta=0.001),
            conv_classifier=L.Convolution2D(64, out_ch, 1, 1, 0, initialW=w)
        )
        self.train = True

    def _upsampling_2d(self, x, pool):
        if x.shape != pool.indexes.shape:
            min_h = min(x.shape[2], pool.indexes.shape[2])
            min_w = min(x.shape[3], pool.indexes.shape[3])
            x = x[:, :, :min_h, :min_w]
            pool.indexes = pool.indexes[:, :, :min_h, :min_w]
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        return F.upsampling_2d(
            x, pool.indexes, ksize=(pool.kh, pool.kw),
            stride=(pool.sy, pool.sx), pad=(pool.ph, pool.pw), outsize=outsize)

    def __call__(self, x):
        p1 = F.MaxPooling2D(2, 2, use_cudnn=False)
        p2 = F.MaxPooling2D(2, 2, use_cudnn=False)
        p3 = F.MaxPooling2D(2, 2, use_cudnn=False)
        p4 = F.MaxPooling2D(2, 2, use_cudnn=False)
        h = F.local_response_normalization(x, 5, 1, 0.0001, 0.75)
        h = p1(F.relu(self.conv1_bn(self.conv1(h), test=not self.train)))
        h = p2(F.relu(self.conv2_bn(self.conv2(h), test=not self.train)))
        h = p3(F.relu(self.conv3_bn(self.conv3(h), test=not self.train)))
        h = p4(F.relu(self.conv4_bn(self.conv4(h), test=not self.train)))
        h = self._upsampling_2d(h, p4)
        h = self.conv_decode4_bn(self.conv_decode4(h), test=not self.train)
        h = self._upsampling_2d(h, p3)
        h = self.conv_decode3_bn(self.conv_decode3(h), test=not self.train)
        h = self._upsampling_2d(h, p2)
        h = self.conv_decode2_bn(self.conv_decode2(h), test=not self.train)
        h = self._upsampling_2d(h, p1)
        h = self.conv_decode1_bn(self.conv_decode1(h), test=not self.train)
        return self.conv_classifier(h)
