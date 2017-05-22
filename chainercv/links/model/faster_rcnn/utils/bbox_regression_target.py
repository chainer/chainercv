from chainer import cuda


def bbox_regression_target(bbox, gt_bbox):
    """Encode bounding boxes into regression targets.

    Given a bounding box, this function computes offsets and scaling to match
    the box to the ground truth box.
    Mathematcially, given a bounding whose center is :math:`p_x, p_y` and size
    :math:`p_w, p_h` and the ground truth bounding box whose center is
    :math:`g_x, g_y` and size :math:`g_w, g_h`, the regression targets
    :math:`t_x, t_y, t_w, t_h` can be computed by the following formulas.

    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`

    The input :obj:`bbox` and :obj:`gt_bbox` are coordinates of bounding boxes,
    whose shapes are :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
    The second axis represents attributes of
    the bounding box. They are :obj:`(x_min, y_min, x_max, y_max)`,
    where the four attributes are coordinates of the bottom left and the
    top right vertices.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [1].

    .. [1] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        bbox (array): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        gt_bbox (array): An array whose shape is :math:`(R, 4)`.
            The shapes of :obj:`bbox` and :obj:`gt_bbox` should be same.

    Returns:
        array:
        Regression targets for mapping from :obj:`bbox` to :obj:`gt_bbox`. \
        This has shape :math:`(R, 4)`. This shape is same as the shape of \
        :obj:`bbox` and :obj:`gt_bbox`.
        The second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    """
    xp = cuda.get_array_module(bbox)

    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]
    ctr_x = bbox[:, 0] + 0.5 * width
    ctr_y = bbox[:, 1] + 0.5 * height

    gt_width = gt_bbox[:, 2] - gt_bbox[:, 0]
    gt_height = gt_bbox[:, 3] - gt_bbox[:, 1]
    gt_ctr_x = gt_bbox[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt_bbox[:, 1] + 0.5 * gt_height

    target_dx = (gt_ctr_x - ctr_x) / width
    target_dy = (gt_ctr_y - ctr_y) / height
    target_dw = xp.log(gt_width / width)
    target_dh = xp.log(gt_height / height)

    target = xp.vstack(
        (target_dx, target_dy, target_dw, target_dh)).transpose()
    return target


def bbox_regression_target_inv(bbox, target):
    """Decode bounding boxes from regression targets.

    Given regression targets computed by :meth:`bbox_regression_target`,
    this function decodes the representation to the bounding box
    representation.

    Given a regression target :math:`t_x, t_y, t_w, t_h` and a bounding
    box whose center is :math:`p_x, p_y` and size :math:`p_w, p_h`,
    the decoded bounding box's center :math:`\\hat{g}_x`, \\hat{g}_y` and size
    :math:`\\hat{g}_w`, :math:`\\hat{g}_h` by the following formulas.

    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`

    The input :obj:`bbox` and the output are coordinates of bounding boxes,
    whose shapes are :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
    The second axis represents attributes of
    the bounding box. They are :obj:`(x_min, y_min, x_max, y_max)`,
    where the four attributes are coordinates of the bottom left and the
    top right vertices.

    The decoding formulas are used in works such as R-CNN [1].

    .. [1] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox (array): An array whose shape is :math:`(R, 4)`.
            Its contents is described above.
        target (array): An array whose shape is :math:`(R, 4)`.
            The shapes of :obj:`bbox` and :obj:`gt_bbox` should be same.
            This contains regression targets, and
            the second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    Returns:
        array:
        Predicted bounding box. Its description is above.

    """
    xp = cuda.get_array_module(bbox)

    if bbox.shape[0] == 0:
        return xp.zeros((0, target.shape[1]), dtype=target.dtype)

    bbox = bbox.astype(target.dtype, copy=False)

    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]
    ctr_x = bbox[:, 0] + 0.5 * width
    ctr_y = bbox[:, 1] + 0.5 * height

    dx = target[:, 0::4]
    dy = target[:, 1::4]
    dw = target[:, 2::4]
    dh = target[:, 3::4]

    pred_ctr_x = dx * width[:, xp.newaxis] + ctr_x[:, xp.newaxis]
    pred_ctr_y = dy * height[:, xp.newaxis] + ctr_y[:, xp.newaxis]
    pred_w = xp.exp(dw) * width[:, xp.newaxis]
    pred_h = xp.exp(dh) * height[:, xp.newaxis]

    pred_bbox = xp.zeros(target.shape, dtype=target.dtype)
    pred_bbox[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_bbox[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # Note that -1 did not exist in the original implementation.
    # However, these subtractions are necessary to make encoding
    # and deconding consistent.
    pred_bbox[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_bbox[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_bbox
