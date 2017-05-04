from chainer import cuda


def bbox_overlap(bbox, query_bbox):
    """Calculate overlap coefficients between bounding boxes.

    The coefficient is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        query_bbox (~numpy.ndarray): An array similar to :obj:`bbox`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        ~numpy.ndarray:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains overlap between \
        :math:`n` th bounding box in :obj:`bbox` and :math:`k` th bounding \
        box in :obj:`query_bbox`.

    """
    if bbox.shape[1] < 4 or query_bbox.shape[1] < 4:
        raise IndexError
    xp = cuda.get_array_module(bbox)

    # left top
    lt = xp.maximum(bbox[:, None, :2], query_bbox[:, :2])
    # right bottom
    rb = xp.minimum(bbox[:, None, 2:4], query_bbox[:, 2:])

    area_i = xp.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_0 = xp.prod(bbox[:, 2:4] - bbox[:, :2], axis=1)
    area_1 = xp.prod(query_bbox[:, 2:4] - query_bbox[:, :2], axis=1)
    return area_i / (area_0[:, None] + area_1 - area_i)
