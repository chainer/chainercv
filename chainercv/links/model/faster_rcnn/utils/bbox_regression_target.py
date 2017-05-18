from chainer import cuda


def delta_encode(raw_bbox, base_raw_bbox):
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
        raw_bbox (array): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        base_raw_bbox (array): An array whose shape is :math:`(R, 4)`.
            The shapes of :obj:`bbox` and :obj:`gt_bbox` should be same.

    Returns:
        array:
        Regression targets for mapping from :obj:`bbox` to :obj:`gt_bbox`. \
        This has shape :math:`(R, 4)`. This shape is same as the shape of \
        :obj:`bbox` and :obj:`gt_bbox`.
        The second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    """
    xp = cuda.get_array_module(raw_bbox)

    width = raw_bbox[:, 2] - raw_bbox[:, 0]
    height = raw_bbox[:, 3] - raw_bbox[:, 1]
    ctr_x = raw_bbox[:, 0] + 0.5 * width
    ctr_y = raw_bbox[:, 1] + 0.5 * height

    base_width = base_raw_bbox[:, 2] - base_raw_bbox[:, 0]
    base_height = base_raw_bbox[:, 3] - base_raw_bbox[:, 1]
    base_ctr_x = base_raw_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = base_raw_bbox[:, 1] + 0.5 * base_height

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = xp.log(base_width / width)
    dh = xp.log(base_height / height)

    bbox = xp.vstack((dx, dy, dw, dh)).transpose()
    return bbox


def delta_decode(base_raw_bbox, bbox):
    """Decode bounding boxes from bounding box offsets.

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
        base_raw_bbox (array): An array whose shape is :math:`(R, 4)`.
            Its contents is described above.
        bbox (array): An array whose shape is :math:`(R, 4)`.
            The shapes of :obj:`bbox` and :obj:`gt_bbox` should be same.
            This contains regression targets, and
            the second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    Returns:
        array:
        Predicted bounding box. Its description is above.

    """
    xp = cuda.get_array_module(bbox)

    if base_raw_bbox.shape[0] == 0:
        return xp.zeros((0, bbox.shape[1]), dtype=bbox.dtype)

    base_raw_bbox = base_raw_bbox.astype(bbox.dtype, copy=False)

    base_width = base_raw_bbox[:, 2] - base_raw_bbox[:, 0]
    base_height = base_raw_bbox[:, 3] - base_raw_bbox[:, 1]
    base_ctr_x = base_raw_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = base_raw_bbox[:, 1] + 0.5 * base_height

    dx = bbox[:, 0::4]
    dy = bbox[:, 1::4]
    dw = bbox[:, 2::4]
    dh = bbox[:, 3::4]

    ctr_x = dx * base_width[:, xp.newaxis] + base_ctr_x[:, xp.newaxis]
    ctr_y = dy * base_height[:, xp.newaxis] + base_ctr_y[:, xp.newaxis]
    w = xp.exp(dw) * base_width[:, xp.newaxis]
    h = xp.exp(dh) * base_height[:, xp.newaxis]

    bbox_raw = xp.zeros(bbox.shape, dtype=bbox.dtype)
    bbox_raw[:, 0::4] = ctr_x - 0.5 * w
    bbox_raw[:, 1::4] = ctr_y - 0.5 * h
    bbox_raw[:, 2::4] = ctr_x + 0.5 * w
    bbox_raw[:, 3::4] = ctr_y + 0.5 * h

    return bbox_raw
