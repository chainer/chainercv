from chainer import cuda


def mask_to_bbox(mask):
    """Compute the bounding boxes around the masked region.

    Args:
        mask (array): An array whose shape is :math:`(N, H, W)`.
            :math:`N` is the number of masks.
            The dtype should be :obj:`numpy.bool`.

    Returns:
        array:
        An array whose shape is :math:`(N, 4)`.
        :math:`N` is the number of bounding boxes.
        The dtype should be :obj:`numpy.float32`.

    """
    _, H, W = mask.shape
    xp = cuda.get_array_module(mask)

    valid_y = xp.any(mask, axis=1)
    y_mins = xp.argmax(mask, axis=1).astype(xp.float32)
    y_mins[xp.logical_not(valid_y)] = xp.inf
    y_min = xp.min(y_mins, axis=1)

    valid_x = xp.any(mask, axis=2)
    x_mins = xp.argmax(mask, axis=2).astype(xp.float32)
    x_mins[xp.logical_not(valid_x)] = xp.inf
    x_min = xp.min(x_mins, axis=1)

    flipped_mask = mask[:, ::-1, ::-1]

    valid_y = xp.any(flipped_mask, axis=1)
    y_maxs = H - xp.argmax(flipped_mask, axis=1).astype(xp.float32)
    y_maxs[xp.logical_not(valid_y)] = -xp.inf
    y_max = xp.max(y_maxs, axis=1)

    valid_x = xp.any(flipped_mask, axis=2)
    x_maxs = W - xp.argmax(flipped_mask, axis=2).astype(xp.float32)
    x_maxs[xp.logical_not(valid_x)] = -xp.inf
    x_max = xp.max(x_maxs, axis=1)
    return xp.stack((y_min, x_min, y_max, x_max), axis=1)
