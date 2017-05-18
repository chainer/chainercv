from chainer import cuda


def delta_encode(raw_bbox_src, raw_bbox_dst):
    """Encodes bounding boxes into deltas of the base bounding boxes.

    Given bounding boxes, this function computes offsets and scales (deltas)
    to match the boxes to the target boxes.
    Mathematcially, given a bounding box whose center is :math:`p_x, p_y` and
    size :math:`p_w, p_h` and the target bounding box whose center is
    :math:`g_x, g_y` and size :math:`g_w, g_h`, the deltas
    :math:`t_x, t_y, t_w, t_h` can be computed by the following formulas.

    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [1].

    .. [1] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        raw_bbox_src (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are used to compute :math:`p_x, p_y, p_w, p_h`.
        raw_bbox_dst (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are used to compute :math:`g_x, g_y, g_w, g_h`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`raw_bbox_src` \
        to :obj:`raw_bbox_dst`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_x, t_y, t_w, t_h`.

    """
    xp = cuda.get_array_module(raw_bbox_src)

    width = raw_bbox_src[:, 2] - raw_bbox_src[:, 0]
    height = raw_bbox_src[:, 3] - raw_bbox_src[:, 1]
    ctr_x = raw_bbox_src[:, 0] + 0.5 * width
    ctr_y = raw_bbox_src[:, 1] + 0.5 * height

    base_width = raw_bbox_dst[:, 2] - raw_bbox_dst[:, 0]
    base_height = raw_bbox_dst[:, 3] - raw_bbox_dst[:, 1]
    base_ctr_x = raw_bbox_dst[:, 0] + 0.5 * base_width
    base_ctr_y = raw_bbox_dst[:, 1] + 0.5 * base_height

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = xp.log(base_width / width)
    dh = xp.log(base_height / height)

    bbox = xp.vstack((dx, dy, dw, dh)).transpose()
    return bbox
