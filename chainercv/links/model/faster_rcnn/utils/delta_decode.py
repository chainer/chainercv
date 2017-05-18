from chainer import cuda


def delta_decode(raw_bbox, bbox):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales (deltas) computed by
    :meth:`delta_encode`, this function decodes the representation to
    coordinates in 2D image space.

    Given a delta :math:`t_x, t_y, t_w, t_h` and a bounding
    box whose center is :math:`p_x, p_y` and size :math:`p_w, p_h`,
    the decoded bounding box's center :math:`\\hat{g}_x`, :math:`\\hat{g}_y`
    and size :math:`\\hat{g}_w`, :math:`\\hat{g}_h` are calculated
    by the following formulas.

    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`

    The decoding formulas are used in works such as R-CNN [1].

    .. [1] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    The output is same type as the type of the inputs.

    Args:
        raw_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are used to
            compute :math:`p_x, p_y, p_w, p_h`.
        bbox (array): An array with offsets and scales.
            The shapes of :obj:`raw_bbox` and :obj:`bbox` should be same.
            This contains values :math:`t_x, t_y, t_w, t_h`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`.

    """
    xp = cuda.get_array_module(bbox)

    if raw_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=bbox.dtype)

    raw_bbox = raw_bbox.astype(raw_bbox.dtype, copy=False)

    base_width = raw_bbox[:, 2] - raw_bbox[:, 0]
    base_height = raw_bbox[:, 3] - raw_bbox[:, 1]
    base_ctr_x = raw_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = raw_bbox[:, 1] + 0.5 * base_height

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
