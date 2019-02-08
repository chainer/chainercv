# Modified work:
# ------------------------------------------------------------------------
# Copyright (c) 2018 Preferred Networks, Inc.
# ------------------------------------------------------------------------

# Original works of CUDA kernel in forward_gpu and forward_gpu:
# ------------------------------------------------------------------------
# Copyright (c) 2017 Microsoft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yi Li, Tairui Chen, Guodong Zhang, Haozhi Qi and Jifeng Dai
# https://github.com/msracver/FCIS
# ------------------------------------------------------------------------


from __future__ import division

import numpy as np
import six

from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check


def _roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(np.floor(size * stride))
    end = int(np.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start


class PSROIAveragePooling2D(function.Function):

    def __init__(self, out_c, out_h, out_w, spatial_scale, group_size):
        if not (isinstance(out_c, int) and out_c > 0):
            raise TypeError(
                'out_c must be positive integer: {}, {}'
                .format(type(out_c), out_c))
        if not (isinstance(out_h, int) and out_h > 0):
            raise TypeError(
                'out_h must be positive integer: {}, {}'
                .format(type(out_h), out_h))
        if not (isinstance(out_w, int) and out_w > 0):
            raise TypeError(
                'out_w must be positive integer: {}, {}'
                .format(type(out_w), out_w))
        if isinstance(spatial_scale, int):
            spatial_scale = float(spatial_scale)
        if not (isinstance(group_size, int) and group_size > 0):
            raise TypeError(
                'group_size must be positive integer: {}, {}'
                .format(type(group_size), group_size))
        self.out_c, self.out_h, self.out_w = out_c, out_h, out_w
        self.spatial_scale = spatial_scale
        self.group_size = group_size

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type, roi_type, roi_index_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim == 4,
            roi_type.dtype == np.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 4,
            roi_index_type.dtype == np.int32,
            roi_index_type.ndim == 1,
            roi_type.shape[0] == roi_index_type.shape[0]
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_roi = bottom_rois.shape[0]
        top_data = np.empty(
            (n_roi, self.out_c, self.out_h, self.out_w), dtype=np.float32)

        for i_roi in six.moves.range(n_roi):
            y_min, x_min, y_max, x_max = bottom_rois[i_roi]
            batch_index = bottom_roi_indices[i_roi]
            y_min = round(y_min * self.spatial_scale)
            x_min = round(x_min * self.spatial_scale)
            y_max = round(y_max * self.spatial_scale)
            x_max = round(x_max * self.spatial_scale)
            roi_height = max(y_max - y_min, 0.1)
            roi_width = max(x_max - x_min, 0.1)

            stride_c = channels / self.out_c
            stride_h = roi_height / self.out_h
            stride_w = roi_width / self.out_w
            group_h = int(round(self.out_h / self.group_size))
            group_w = int(round(self.out_w / self.group_size))

            for out_h in six.moves.range(self.out_h):
                slice_h, len_h = _roi_pooling_slice(
                    out_h, stride_h, height, int(y_min))
                if slice_h.stop <= slice_h.start:
                    continue
                for out_w in six.moves.range(self.out_w):
                    slice_w, len_w = _roi_pooling_slice(
                        out_w, stride_w, width, int(x_min))
                    if slice_w.stop <= slice_w.start:
                        continue
                    for out_c in six.moves.range(self.out_c):
                        slice_c, len_c = _roi_pooling_slice(
                            out_c, stride_c, channels, 0)
                        roi_data = bottom_data[
                            batch_index, slice_c, slice_h, slice_w]\
                            .reshape((len_c, -1))
                        c = (out_h // group_h) * self.group_size \
                            + (out_w // group_w)
                        top_data[i_roi, out_c, out_h, out_w] = np.average(
                            roi_data[c])
        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_roi = bottom_rois.shape[0]
        top_data = cuda.cupy.empty(
            (n_roi, self.out_c, self.out_h, self.out_w), dtype=np.float32)
        cuda.elementwise(
            '''
            raw float32 bottom_data, raw float32 bottom_rois,
            raw int32 bottom_roi_indices,
            float32 spatial_scale, int32 channels,
            int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size
            ''',
            'float32 top_data',
            '''
            // pos in output filter
            int ph = (i / pooled_width) % pooled_height;
            int pw = i % pooled_width;
            int ctop = (i / pooled_width / pooled_height) % pooled_dim;
            int n = i / pooled_width / pooled_height / pooled_dim;

            int roi_batch_ind = bottom_roi_indices[n];
            float roi_start_h = static_cast<float>(
                round(bottom_rois[n * 4 + 0])) * spatial_scale;
            float roi_start_w = static_cast<float>(
                round(bottom_rois[n * 4 + 1])) * spatial_scale;
            float roi_end_h = static_cast<float>(
                round(bottom_rois[n * 4 + 2])) * spatial_scale;
            float roi_end_w = static_cast<float>(
                round(bottom_rois[n * 4 + 3])) * spatial_scale;

            // Force too small ROIs to be 1x1
            float roi_height = max(roi_end_h - roi_start_h, 0.1);
            float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

            // Compute w and h at bottom
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                                * bin_size_h + roi_start_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                                * bin_size_w + roi_start_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                             * bin_size_h + roi_start_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                            * bin_size_w + roi_start_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height);
            wstart = min(max(wstart, 0), width);
            hend = min(max(hend, 0), height);
            wend = min(max(wend, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Compute c at bottom
            int gh = floor(
                static_cast<float>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<float>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int data_offset = (roi_batch_ind * channels + c) * height * width;
            float out_sum = 0;
            for (int h = hstart; h < hend; ++h){
              for (int w = wstart; w < wend; ++w){
                 int bottom_index = h * width + w;
                 out_sum += bottom_data[data_offset + bottom_index];
              }
            }

            float bin_area = (hend - hstart) * (wend - wstart);
            top_data = is_empty? (float) 0. : out_sum / bin_area;
            ''', 'ps_roi_average_pooling_2d_fwd'
        )(bottom_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          self.out_c, self.out_h, self.out_w, self.group_size,
          top_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
        n_roi = bottom_rois.shape[0]
        bottom_diff = np.zeros(self._bottom_data_shape, np.float32)

        for i_roi in six.moves.range(n_roi):
            y_min, x_min, y_max, x_max = bottom_rois[i_roi]
            batch_index = bottom_roi_indices[i_roi]
            y_min = round(y_min * self.spatial_scale)
            x_min = round(x_min * self.spatial_scale)
            y_max = round(y_max * self.spatial_scale)
            x_max = round(x_max * self.spatial_scale)
            roi_height = max(y_max - y_min, 0.1)
            roi_width = max(x_max - x_min, 0.1)

            stride_c = channels / self.out_c
            stride_h = roi_height / self.out_h
            stride_w = roi_width / self.out_w
            group_h = int(round(self.out_h / self.group_size))
            group_w = int(round(self.out_w / self.group_size))

            for out_h in six.moves.range(self.out_h):
                slice_h, len_h = _roi_pooling_slice(
                    out_h, stride_h, height, int(y_min))
                if slice_h.stop <= slice_h.start:
                    continue
                for out_w in six.moves.range(self.out_w):
                    slice_w, len_w = _roi_pooling_slice(
                        out_w, stride_w, width, int(x_min))
                    if slice_w.stop <= slice_w.start:
                        continue
                    for out_c in six.moves.range(self.out_c):
                        diff_val = gy[0][i_roi, out_c, out_h, out_w]
                        diff_val = diff_val / len_h / len_w
                        start_c = int(np.floor(out_c * stride_c))
                        start_c = min(max(start_c, 0), channels)

                        c = (out_h // group_h) * self.group_size \
                            + (out_w // group_w) + start_c
                        bottom_diff[batch_index, c, slice_h, slice_w] \
                            += diff_val
        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, np.float32)
        cuda.elementwise(
            '''
            raw float32 top_diff, raw float32 bottom_rois,
            raw int32 bottom_roi_indices,
            float32 spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size
            ''',
            'raw float32 bottom_diff',
            '''
            int ph = (i / pooled_width) % pooled_height;
            int pw = i % pooled_width;
            int ctop = (i / pooled_width / pooled_height) % pooled_dim;
            int n = i / pooled_width / pooled_height / pooled_dim;

            // [start, end) interval for spatial sampling
            int roi_batch_ind = bottom_roi_indices[n];
            float roi_start_h = static_cast<float>(
                round(bottom_rois[n * 4 + 0])) * spatial_scale;
            float roi_start_w = static_cast<float>(
                round(bottom_rois[n * 4 + 1])) * spatial_scale;
            float roi_end_h = static_cast<float>(
                round(bottom_rois[n * 4 + 2])) * spatial_scale;
            float roi_end_w = static_cast<float>(
                round(bottom_rois[n * 4 + 3])) * spatial_scale;

            // Force too small ROIs to be 1x1
            float roi_height = max(roi_end_h - roi_start_h, 0.1);
            float roi_width = max(roi_end_w - roi_start_w, 0.1); // avoid 0

            // Compute w and h at bottom
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);

            int hstart = floor(
                static_cast<float>(ph) * bin_size_h + roi_start_h);
            int wstart = floor(
                static_cast<float>(pw) * bin_size_w + roi_start_w);
            int hend = ceil(
                static_cast<float>(ph + 1.0) * bin_size_h + roi_start_h);
            int wend = ceil(
                static_cast<float>(pw + 1.0) * bin_size_w + roi_start_w);

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height);
            wstart = min(max(wstart, 0), width);
            hend = min(max(hend, 0), height);
            wend = min(max(wend, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Compute c at bottom
            int gh = floor(
                static_cast<float>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<float>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int bottom_diff_offset = (roi_batch_ind * channels + c);
            bottom_diff_offset = bottom_diff_offset * height * width;
            int top_offset =
                (n * pooled_dim + ctop) * pooled_height * pooled_width;

            float bin_area = (hend - hstart) * (wend - wstart);
            float diff_val = is_empty ? (float) 0. :
                top_diff[top_offset + ph * pooled_width + pw] / bin_area;
            for (int h = hstart; h < hend; ++h){
              for (int w = wstart; w < wend; ++w){
                int bottom_index = h * width + w;
                atomicAdd(
                    &bottom_diff[bottom_diff_offset + bottom_index], diff_val);
              }
            }
            ''', 'ps_roi_average_pooling_2d_bwd'
        )(gy[0], bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          self.out_c, self.out_h, self.out_w,
          self.group_size, bottom_diff, size=gy[0].size)

        return bottom_diff, None, None


def ps_roi_average_pooling_2d(
        x, rois, roi_indices, out_c, out_h, out_w,
        spatial_scale, group_size
):
    """Position Sensitive Region of Interest (ROI) Average pooling function.

    This function computes position sensitive average of input spatial patch
    with the given region of interests. Each ROI is splitted into
    :math:`(group\_size, group\_size)` regions, and position sensitive values
    in each region is computed.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (array): Input roi. The shape is expected to
            be :math:`(R, 4)`, and each datum is set as below:
            (y_min, x_min, y_max, x_max). The dtype is :obj:`numpy.float32`.
        roi_indices (array): Input roi indices. The shape is expected to
            be :math:`(R, )`. The dtype is :obj:`numpy.int32`.
        out_c (int): Channels of output image after pooled.
        out_h (int): Height of output image after pooled.
        out_w (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.
        group_size (int): Position sensitive group size.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing PSROIPooling:
    `R-FCN <https://arxiv.org/abs/1605.06409>`_.

    """
    return PSROIAveragePooling2D(out_c, out_h, out_w, spatial_scale,
                                 group_size)(x, rois, roi_indices)
