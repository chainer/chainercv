import chainer
import chainer.functions as F
import chainer.links as L


class Encoder(chainer.Chain):

    def __init__(self, in_ch, out_ch, n_conv=3):
        w = chainer.initializers.HeNormal()
        super(Encoder, self).__init__(
            conv1=L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w),
            conv2=L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w),
            bn1=L.BatchNormalization(out_ch, initial_beta=0.001),
            bn2=L.BatchNormalization(out_ch, initial_beta=0.001),
        )
        if n_conv == 3:
            self.add_link(
                'conv3', L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w))
            self.add_link(
                'bn3', L.BatchNormalization(out_ch, initial_beta=0.001))
        self.pool = F.MaxPooling2D(2, 2, use_cudnn=False)
        self.n_conv = n_conv

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = F.relu(self.bn2(self.conv2(h), test=not train))
        if self.n_conv == 3:
            h = F.relu(self.bn3(self.conv3(h), test=not train))
        return self.pool(h)


class Decoder(chainer.Chain):

    def __init__(self, in_ch, out_ch, n_conv=3):
        w = chainer.initializers.HeNormal()
        mid_ch = in_ch if n_conv == 3 else out_ch
        super(Decoder, self).__init__(
            conv1=L.Convolution2D(in_ch, in_ch, 3, 1, 1, initialW=w),
            conv2=L.Convolution2D(in_ch, mid_ch, 3, 1, 1, initialW=w),
            bn1=L.BatchNormalization(in_ch, initial_beta=0.001),
            bn2=L.BatchNormalization(mid_ch, initial_beta=0.001),
        )
        if n_conv == 3:
            self.add_link(
                'conv3', L.Convolution2D(mid_ch, out_ch, 3, 1, 1, initialW=w))
            self.add_link(
                'bn3', L.BatchNormalization(out_ch, initial_beta=0.001))
        self.n_conv = n_conv

    def _upsampling_2d(self, x, pool):
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        return F.upsampling_2d(
            x, pool.indexes, ksize=(pool.kh, pool.kw),
            stride=(pool.sy, pool.sx), pad=(pool.ph, pool.pw), outsize=outsize)

    def __call__(self, x, pool, train):
        h = self._upsampling_2d(x, pool)
        h = F.relu(self.bn1(self.conv1(h), test=not train))
        h = F.relu(self.bn2(self.conv2(h), test=not train))
        if self.n_conv == 3:
            h = F.relu(self.bn3(self.conv3(h), test=not train))
        return h


class SegNet(chainer.Chain):

    def __init__(self, out_ch):
        super(SegNet, self).__init__(
            enc1=Encoder(None, 64, n_conv=2),
            enc2=Encoder(64, 128, n_conv=2),
            enc3=Encoder(128, 256),
            enc4=Encoder(256, 512),
            enc5=Encoder(512, 512),
            dec5=Decoder(512, 512),
            dec4=Decoder(512, 256),
            dec3=Decoder(256, 128),
            dec2=Decoder(128, 64, n_conv=2),
            dec1=Decoder(64, out_ch, n_conv=2),
        )
        self.train = True

    def __call__(self, x):
        h = self.enc1(x, self.train)
        h = self.enc2(h, self.train)
        h = self.enc3(h, self.train)
        h = self.enc4(h, self.train)
        h = self.enc5(h, self.train)
        h = self.dec5(h, self.enc5.pool, self.train)
        h = self.dec4(h, self.enc4.pool, self.train)
        h = self.dec3(h, self.enc3.pool, self.train)
        h = self.dec2(h, self.enc2.pool, self.train)
        return self.dec1(h, self.enc1.pool, self.train)
