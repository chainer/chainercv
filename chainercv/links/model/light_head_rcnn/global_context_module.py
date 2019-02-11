import chainer
import chainer.functions as F
import chainer.links as L


class GlobalContextModule(chainer.Chain):

    def __init__(
            self, in_channels, mid_channels, out_channels,
            ksize, initialW=None
    ):
        super(GlobalContextModule, self).__init__()
        with self.init_scope():
            padsize = int((ksize - 1) / 2)
            self.col_max = L.Convolution2D(
                in_channels, mid_channels, (ksize, 1), 1, (padsize, 1),
                initialW=initialW)
            self.col = L.Convolution2D(
                mid_channels, out_channels, (1, ksize), 1, (1, padsize),
                initialW=initialW)
            self.row_max = L.Convolution2D(
                in_channels, mid_channels, (1, ksize), 1, (1, padsize),
                initialW=initialW)
            self.row = L.Convolution2D(
                mid_channels, out_channels, (ksize, 1), 1, (padsize, 1),
                initialW=initialW)

    def __call__(self, x):
        h_col = self.col(self.col_max(x))
        h_row = self.row(self.row_max(x))
        return F.relu(h_col + h_row)
