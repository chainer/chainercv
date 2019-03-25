import chainer
from chainer.backends import cuda

from chainercv import transforms


class YOLOBase(chainer.Chain):
    """Base class for YOLOv2 and YOLOv3.

    A subclass of this class should have :obj:`extractor`,
    :meth:`forward`, and :meth:`_decode`.
    """

    @property
    def insize(self):
        return self.extractor.insize

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`nms_thresh` and
        :obj:`score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'}): A string to determine the
                preset to use.
        """

        if preset == 'visualize':
            self.nms_thresh = 0.45
            self.score_thresh = 0.5
        elif preset == 'evaluate':
            self.nms_thresh = 0.45
            self.score_thresh = 0.005
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bounding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """

        x = []
        params = []
        for img in imgs:
            _, H, W = img.shape
            img, param = transforms.resize_contain(
                img / 255, (self.insize, self.insize), fill=0.5,
                return_param=True)
            x.append(self.xp.array(img))
            param['size'] = (H, W)
            params.append(param)

        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            locs, objs, confs = self.forward(self.xp.stack(x))
        locs = locs.array
        objs = objs.array
        confs = confs.array

        bboxes = []
        labels = []
        scores = []
        for loc, obj, conf, param in zip(locs, objs, confs, params):
            bbox, label, score = self._decode(loc, obj, conf)
            bbox = cuda.to_cpu(bbox)
            label = cuda.to_cpu(label)
            score = cuda.to_cpu(score)

            bbox = transforms.translate_bbox(
                bbox, -self.insize / 2, -self.insize / 2)
            bbox = transforms.resize_bbox(
                bbox, param['scaled_size'], param['size'])
            bbox = transforms.translate_bbox(
                bbox, param['size'][0] / 2, param['size'][1] / 2)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores
