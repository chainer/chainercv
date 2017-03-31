from __future__ import division
import numpy as np


def non_maximum_suppression(bbox, threshold, limit=None, return_param=False):
    """Suppress bounding boxes according to their Jaccard overlap.

    This method checks each bounding box sequentially and selects the box
    if the Jaccard overlap between the box and previously selected boxes
    is less than :obj:`threshold`. This method is mainly used as postprocessing
    of object detection. In this case, the bounding boxes should be sorted in
    descending order.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :obj:`(x_min, y_min, x_max, y_max)`,
    where the four attributes are coordinates of the bottom left and the
    top right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be augmented. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        threshold (float): Thresold of Jaccard overlap.
        limit (int): the upper bound of the number of selected boxes. If it is
            not specified, this method selects as many boxes as possible.
        return_param (bool): returns information of selection.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        If :obj:`return_param = False`,
        returns an array :obj:`out_bbox` that is the result of suppression.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **selection** (:obj:`~numpy.ndarray`): An array which indicates each\
            bounding box is selected or not.

    """

    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selection = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        lt = np.maximum(b[:2], bbox[selection, :2])
        rb = np.minimum(b[2:], bbox[selection, 2:])
        area = np.prod(rb - lt, axis=1) * (lt < rb).all(axis=1)

        jaccard = area / (bbox_area[i] + bbox_area[selection] - area)
        if (jaccard >= threshold).any():
            continue

        selection[i] = True
        if limit and np.count_nonzero(selection) >= limit:
            break

    if return_param:
        return bbox[selection], {'selection': selection}
    else:
        return bbox[selection]
