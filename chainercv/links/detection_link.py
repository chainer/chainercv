import chainer


class DetectionLink(chainer.Link):
    """A chainer.Link for object detection.

    This is an abstract class for object detection links.
    All object detectors should inherit this class.
    """

    def predict(self, img):
        """Detect objects in an image.

        This method detects objects in an image.
        This method returns a tuple of three arrays,
        :obj:`(bbox, label, score)`.

        Args:
            img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
                This is in BGR format and the range of its value is
                :math:`[0, 255]`.

        Returns:
            (~numpy.ndarray, ~numpy.ndarray, ~numpy.ndarray):
            This method returns a tuple of three arrays,
            :obj:`(bbox, label, score)`.

            * **bbox**: A float array of shape :math:`(R, 4)`, where \
                 :math:`R` is the number of bounding boxes in the image. \
                Elements are organized by :obj:`(x_min, y_min, x_max, y_max)` \
                in the second axis.
            * **label** : An integer array of shape :math:`(R,)`. \
                Each value indicates the class of the bounding box.
            * **score** : A float array of shape :math:`(R,)`. \
                Each value indicates how confident the prediction is.
        """

        raise NotImplementedError
