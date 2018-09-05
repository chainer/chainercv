import numpy as np

import chainer
from chainer.backends import cuda

from chainercv import transforms


class FasterRCNN(chainer.Chain):

    _min_size = 800
    _max_size = 1333
    _stride = 32

    def __init__(self, extractor, rpn, head):
        super().__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.use_preset('visualize')

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.5
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def __call__(self, x):
        assert(not chainer.config.train)
        hs = self.extractor(x)
        rpn_locs, rpn_confs = self.rpn(hs)
        anchors = self.rpn.anchors(h.shape[2:] for h in hs)
        rois, roi_indices = self.rpn.decode(
            rpn_locs, rpn_confs, anchors, x.shape)
        rois, roi_indices = self.head.distribute(rois, roi_indices)
        head_locs, head_confs = self.head(hs, rois, roi_indices)
        return rois, roi_indices, head_locs, head_confs

    def predict(self, imgs):
        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            rois, roi_indices, head_locs, head_confs = self(x)
        bboxes, labels, scores = self.head.decode(
            rois, roi_indices, head_locs, head_confs,
            scales, sizes, self.nms_thresh, self.score_thresh)

        bboxes = [cuda.to_cpu(bbox) for bbox in bboxes]
        labels = [cuda.to_cpu(label) for label in labels]
        scores = [cuda.to_cpu(score) for score in scores]
        return bboxes, labels, scores

    def prepare(self, imgs):
        scales = []
        resized_imgs = []
        for img in imgs:
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            scales.append(scale)
            H, W = int(H * scale), int(W * scale)
            img = transforms.resize(img, (H, W))
            img -= self.extractor.mean
            resized_imgs.append(img)

        size = np.array([img.shape[1:] for img in resized_imgs]).max(axis=0)
        size = (np.ceil(size / self._stride) * self._stride).astype(int)
        x = np.zeros((len(imgs), 3, size[0], size[1]), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img

        x = self.xp.array(x)
        return x, scales
