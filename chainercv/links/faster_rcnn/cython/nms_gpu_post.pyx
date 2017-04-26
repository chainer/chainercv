cimport numpy as np
from libc.stdint cimport uint64_t

import numpy as np


def nms_gpu_post(np.ndarray[np.uint64_t, ndim=1] mask,
                 int boxes_num,
                 int threads_per_block,
                 int col_blocks
                 ):
    cdef int i, j, nblock, index
    cdef uint64_t inblock
    cdef int num_to_keep = 0
    cdef uint64_t one_ull = 1
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.uint64_t, ndim=1] remv = np.zeros((col_blocks,), dtype=np.uint64)

    for i in range(boxes_num):
        nblock = i / threads_per_block
        inblock = i % threads_per_block

        if not (remv[nblock] & one_ull << inblock):
            keep[num_to_keep] = i
            num_to_keep += 1

            index = i * col_blocks
            for j in range(nblock, col_blocks):
                remv[j] |= mask[index + j]
    return keep, num_to_keep
