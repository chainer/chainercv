from __future__ import division

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.faster_rcnn.utils.bbox_regression_target import \
    bbox_regression_target_inv
from chainercv.utils.bbox import non_maximum_suppression

from chainercv.transforms.image.resize import resize


class FasterRCNNBase(chainer.Chain):

    def __init__(
            self, feature, rpn, head,
            n_class, roi_size,
            nms_thresh=0.3,
            score_thresh=0.7,
            spatial_scale=0.0625,
            bbox_normalize_mean=(0., 0., 0., 0.),
            bbox_normalize_std=(0.1, 0.1, 0.2, 0.2),
            proposal_target_layer_params={},
    ):
        super(FasterRCNNBase, self).__init__(
            feature=feature,
            rpn=rpn,
            head=head,
        )
        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.bbox_normalize_mean = bbox_normalize_mean
        self.bbox_normalize_std = bbox_normalize_std

        self.mean = np.array([102.9801, 115.9465, 122.7717])[:, None, None]
        self.min_size = 600
        self.max_size = 1000

    def _decide_when_to_stop(self, layers):
        if len(layers):
            return 'start'

        rpn_outs = [
            'feature', 'rpn_bbox_pred', 'rpn_cls_score',
            'roi', 'anchor']
        for layer in rpn_outs:
            layers.pop(layer, None)

        if len(layers) == 0:
            return 'rpn'
        return 'head'

    def __call__(self, x, scale=1., layers=['bbox', 'cls_prob'],
                 test=True):
        activations = {key: None for key in layers}
        stop_at = self._decide_when_to_stop(activations)
        if stop_at == 'start':
            return {}

        if isinstance(scale, chainer.Variable):
            scale = scale.data
        if isinstance(scale, float):
            scale = np.array(scale)
        scale = np.asscalar(cuda.to_cpu(scale))
        img_size = x.shape[2:][::-1]

        h = self.feature(x, train=not test)
        rpn_bbox_pred, rpn_cls_score, roi, anchor =\
            self.rpn(h, img_size, scale, train=not test)

        _update_if_specified({
            'feature': h,
            'rpn_bbox_pred': rpn_bbox_pred,
            'rpn_cls_score': rpn_cls_score,
            'roi': roi,
            'anchor': anchor
        }, activations)
        if stop_at == 'rpn':
            return activations

        pool = F.roi_pooling_2d(
            h, roi, self.roi_size, self.roi_size, self.spatial_scale)
        head_out = self.head(pool, train=False)
        _update_if_specified(head_out, activations)
        _update_if_specified({'pool': pool}, activations)
        return activations

    def _suppress(self, raw_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for i in range(1, self.n_class):
            bbox_cls = raw_bbox[:, i * 4: (i + 1) * 4]
            prob_cls = raw_prob[:, i]
            mask = prob_cls > self.score_thresh
            bbox_cls = bbox_cls[mask]
            prob_cls = prob_cls[mask]
            keep = non_maximum_suppression(
                bbox_cls, self.nms_thresh, prob_cls)

            bbox.append(bbox_cls[keep])
            label.append(i * np.ones((len(keep),)))
            score.append(prob_cls[keep])
        bbox = self.xp.asarray(np.concatenate(bbox, axis=0).astype(np.float32))
        label = self.xp.asarray(np.concatenate(label, axis=0).astype(np.int32))
        score = self.xp.asarray(
            np.concatenate(score, axis=0).astype(np.float32))
        return bbox, label, score

    def predict(self, imgs):
        """Detect objects from images

        This method predicts objects for each image.

        Args:
            imgs (iterable of ~numpy.ndarray): Arrays holding images.
                All images are in CHW and BGR format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of list:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :obj:`(x_min, y_min, x_max, y_max)` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        prepared_imgs = list()
        scales = list()
        for img in imgs:
            _, H, W = img.shape
            img, scale = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            scales.append(scale)

        bboxes = list()
        labels = list()
        scores = list()
        for img, scale in zip(prepared_imgs, scales):
            img_var = chainer.Variable(
                self.xp.asarray(img[None]), volatile=chainer.flag.ON)
            H, W = img_var.shape[2:]
            out = self.__call__(
                img_var, scale=scale, layers=['roi', 'bbox_tf', 'score'])

            roi = out['roi']
            bbox_tf = out['bbox_tf']
            score = out['score']
            xp = chainer.cuda.get_array_module(bbox_tf)
            # Convert predictions to bounding boxes in image coordinates.
            bbox_roi = roi[:, 1:5]
            bbox_roi = bbox_roi / scale
            bbox_tf_data = bbox_tf.data

            mean = xp.tile(xp.array(self.bbox_normalize_mean), self.n_class)
            std = xp.tile(np.array(self.bbox_normalize_std), self.n_class)
            bbox_tf_data = (bbox_tf_data * std + mean).astype(np.float32)
            raw_bbox = bbox_regression_target_inv(bbox_roi, bbox_tf_data)

            # clip bounding box
            raw_bbox[:, slice(0, 4, 2)] = xp.clip(
                raw_bbox[:, slice(0, 4, 2)], 0, W / scale)
            raw_bbox[:, slice(1, 4, 2)] = xp.clip(
                raw_bbox[:, slice(1, 4, 2)], 0, H / scale)
            # Compute probabilities that each bounding box is assigned to.
            raw_prob = F.softmax(score).data

            bbox, label, score = self._suppress(raw_bbox, raw_prob)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores

    def prepare(self, img):
        """Preprocess an image for feature extraction.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and BGR format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.
        """
        _, H, W = img.shape

        scale = 1.
        if min(H, W) < self.min_size:
            scale = self.min_size / min(H, W)

        if scale * max(H, W) > self.max_size:
            scale = max(H, W) * scale / self.max_size

        img = resize(img, (W * scale, H * scale))

        img -= self.mean
        return img, scale


def _update_if_specified(source, target):
    for key in source.keys():
        if key in target:
            target[key] = source[key]
