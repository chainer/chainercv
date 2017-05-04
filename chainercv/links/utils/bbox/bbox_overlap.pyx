# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

cimport cython

import numpy as np
cimport numpy as np


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
def bbox_overlap(
        np.ndarray[DTYPE_t, ndim=2] bbox,
        np.ndarray[DTYPE_t, ndim=2] query_bbox):
    """Calculate overlap coefficients between bounding boxes.

    The coefficient is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        query_bbox (~numpy.ndarray): An array similar to :obj:`bbox`,
            whose shape is :math:`(K, 4)`. The dtype should be :obj:`numpy.float32`.

    Returns:
        ~numpy.ndarray:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains overlap between \
        :math:`n` th bounding box in :obj:`bbox` and :math:`k` th bounding \
        box in :obj:`query_bbox`.

    """
    cdef:
        unsigned int k, n, N, K
        DTYPE_t iw, ih, box_area
        DTYPE_t ua
        np.ndarray[DTYPE_t, ndim=2] overlap

    N = bbox.shape[0]
    K = query_bbox.shape[0]
    if bbox.shape[1] < 4 or query_bbox.shape[1] < 4:
        raise IndexError

    overlap  = np.zeros((N, K), dtype=DTYPE)

    for k in range(K):
        box_area = (
            (query_bbox[k, 2] - query_bbox[k, 0] + 1) *
            (query_bbox[k, 3] - query_bbox[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(bbox[n, 2], query_bbox[k, 2]) -
                max(bbox[n, 0], query_bbox[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(bbox[n, 3], query_bbox[k, 3]) -
                    max(bbox[n, 1], query_bbox[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (bbox[n, 2] - bbox[n, 0] + 1) *
                        (bbox[n, 3] - bbox[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlap[n, k] = iw * ih / ua
    return overlap
