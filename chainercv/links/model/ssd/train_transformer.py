import numpy as np

from chainercv.links.model.ssd import encode_with_default_bbox
from chainercv import transforms


class TrainTransformer(object):

    def __init__(self, insize, mean, default_bbox, variance):
        self.insize = insize
        self.mean = mean
        self.default_bbox = default_bbox
        self.variance = variance

    def __call__(self, in_data):
        img, bbox, label = in_data

        _, H, W = img.shape
        x = transforms.resize(img, (self.insize, self.insize))
        x -= np.array(self.mean)[:, np.newaxis, np.newaxis]
        bbox = bbox.copy()
        bbox[:, 0::2] /= W
        bbox[:, 1::2] /= H
        loc, conf = encode_with_default_bbox(
            bbox, label,
            self.default_bbox, self.variance, 0.5)

        return x, loc, conf
