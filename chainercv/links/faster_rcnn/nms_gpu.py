import numpy as np
import cupy

from nms_gpu_post import nms_gpu_post


code = '''
#include <stdio.h>
#include <vector>
#include <iostream>

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

extern "C"
__global__
void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
'''


def _nms(boxes_host, boxes_num, boxes_dim, nms_overlap_thresh):

    threads_per_block = 64

    col_blocks = divup(boxes_num, threads_per_block)

    boxes_dev = cupy.array(boxes_host, dtype=np.float32)
    # boxes_dev = cupy.asfortranarray(boxes_dev)
    mask_dev = cupy.zeros((boxes_num * col_blocks,), dtype=np.uint64)

    blocks = (divup(boxes_num, threads_per_block), divup(boxes_num, threads_per_block), 1)
    threads = (threads_per_block, 1, 1)

    kern = load_kernel('nms_kernel', code)
    kern(blocks, threads, args=(boxes_num, cupy.float32(nms_overlap_thresh),
                                boxes_dev, mask_dev))

    mask_host = mask_dev.get()

    keep, num_to_keep = nms_gpu_post(
        mask_host, boxes_num, threads_per_block, col_blocks)
    return keep, num_to_keep


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, options=()):
    assert isinstance(options, tuple)
    kernel_code = cupy.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)


def nms_gpu(dets, thresh):
    boxes_num = dets.shape[0]
    boxes_dim = dets.shape[1]

    scores = dets[:, 4]
    order = scores.argsort()[::-1]
    sorted_dets = dets[order, :]
    keep, num_out = _nms(keep, sorted_dets, boxes_num, boxes_dim, thresh)
    # _nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    return list(order[keep])


def divup(m, n):
    return m / n + int(m % n > 0)
