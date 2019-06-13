# Modified work:
# ------------------------------------------------------------------------
# Copyright (c) 2018 Preferred Networks, Inc.
# ------------------------------------------------------------------------

# Original works of CUDA kernel in forward_gpu and backward_gpu:
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

import numbers
import numpy as np
import six

from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check

from chainercv.functions.ps_roi_average_pooling_2d import _outsize


class PSROIMaxPooling2D(function.Function):

    def __init__(self, outsize, spatial_scale, group_size):
        out_c, out_h, out_w = _outsize(outsize)
        if out_c is not None and \
                not (isinstance(out_c, numbers.Integral) and out_c > 0):
            raise TypeError(
                'outsize[0] must be positive integer: {}, {}'
                .format(type(out_c), out_c))
        if not (isinstance(out_h, numbers.Integral) and out_h > 0):
            raise TypeError(
                'outsize[1] must be positive integer: {}, {}'
                .format(type(out_h), out_h))
        if not (isinstance(out_w, numbers.Integral) and out_w > 0):
            raise TypeError(
                'outsize[2] must be positive integer: {}, {}'
                .format(type(out_w), out_w))
        if isinstance(spatial_scale, numbers.Integral):
            spatial_scale = float(spatial_scale)
        if not (isinstance(spatial_scale, numbers.Real)
                and spatial_scale > 0):
            raise TypeError(
                'spatial_scale must be a positive float number: {}, {}'
                .format(type(spatial_scale), spatial_scale))
        if not (isinstance(group_size, numbers.Integral) and group_size > 0):
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
        channel, height, width = bottom_data.shape[1:]
        if self.out_c is None:
            if channel % (self.group_size * self.group_size) != 0:
                raise ValueError(
                    'input channel must be divided by group_size * group_size:'
                    '{} % {} != 0'
                    .format(channel, self.group_size * self.group_size))
            out_c = channel // (self.group_size * self.group_size)
        else:
            if channel != self.out_c * self.group_size * self.group_size:
                raise ValueError(
                    'input channel must be equal to '
                    'outsize[0] * group_size * group_size: {} != {}'
                    .format(channel,
                            self.out_c * self.group_size * self.group_size))
            out_c = self.out_c
        n_roi = bottom_rois.shape[0]
        top_data = np.empty(
            (n_roi, out_c, self.out_h, self.out_w), dtype=np.float32)
        self.argmax_data = np.empty(top_data.shape, dtype=np.int32)

        spatial_scale = self.spatial_scale
        pooled_height = self.out_h
        pooled_width = self.out_w
        group_size = self.group_size

        for i in six.moves.range(top_data.size):
            n, ctop, ph, pw = np.unravel_index(i, top_data.shape)

            roi_batch_ind = bottom_roi_indices[n]
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 0.1)
            roi_width = max(roi_end_w - roi_start_w, 0.1)
            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            hstart = int(np.floor(ph * bin_size_h + roi_start_h))
            wstart = int(np.floor(pw * bin_size_w + roi_start_w))
            hend = int(np.ceil((ph + 1) * bin_size_h + roi_start_h))
            wend = int(np.ceil((pw + 1) * bin_size_w + roi_start_w))
            hstart = min(max(hstart, 0), height)
            wstart = min(max(wstart, 0), width)
            hend = min(max(hend, 0), height)
            wend = min(max(wend, 0), width)

            gh = int(np.floor(ph * group_size / pooled_height))
            gw = int(np.floor(pw * group_size / pooled_width))
            gh = min(max(gh, 0), group_size - 1)
            gw = min(max(gw, 0), group_size - 1)
            c = (ctop * group_size + gh) * group_size + gw

            maxval = - np.inf
            maxidx = -1
            for y in six.moves.range(hstart, hend):
                for x in six.moves.range(wstart, wend):
                    tmpval = bottom_data[roi_batch_ind, c, y, x]
                    bottom_index = y * width + x
                    if (tmpval > maxval):
                        maxval = tmpval
                        maxidx = bottom_index
            top_data[n, ctop, ph, pw] = maxval
            self.argmax_data[n, ctop, ph, pw] = maxidx

        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channel, height, width = bottom_data.shape[1:]
        if self.out_c is None:
            if channel % (self.group_size * self.group_size) != 0:
                raise ValueError(
                    'input channel must be divided by group_size * group_size:'
                    '{} % {} != 0'
                    .format(channel, self.group_size * self.group_size))
            out_c = channel // (self.group_size * self.group_size)
        else:
            if channel != self.out_c * self.group_size * self.group_size:
                raise ValueError(
                    'input channel must be equal to '
                    'outsize[0] * group_size * group_size: {} != {}'
                    .format(channel,
                            self.out_c * self.group_size * self.group_size))
            out_c = self.out_c
        n_roi = bottom_rois.shape[0]
        top_data = cuda.cupy.empty(
            (n_roi, out_c, self.out_h, self.out_w), dtype=np.float32)
        self.argmax_data = cuda.cupy.empty(top_data.shape, np.int32)

        cuda.elementwise(
            '''
            raw T bottom_data, raw T bottom_rois,
            raw int32 bottom_roi_indices,
            T spatial_scale, int32 channels,
            int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size
            ''',
            'T top_data, int32 argmax_data',
            '''
            // pos in output filter
            int ph = (i / pooled_width) % pooled_height;
            int pw = i % pooled_width;
            int ctop = (i / pooled_width / pooled_height) % pooled_dim;
            int n = i / pooled_width / pooled_height / pooled_dim;

            int roi_batch_ind = bottom_roi_indices[n];
            T roi_start_h = bottom_rois[n * 4 + 0] * spatial_scale;
            T roi_start_w = bottom_rois[n * 4 + 1] * spatial_scale;
            T roi_end_h = bottom_rois[n * 4 + 2] * spatial_scale;
            T roi_end_w = bottom_rois[n * 4 + 3] * spatial_scale;

            // Force too small ROIs to be 1x1
            T roi_height = max(roi_end_h - roi_start_h, 0.1);
            T roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

            // Compute w and h at bottom
            T bin_size_h = roi_height / static_cast<T>(pooled_height);
            T bin_size_w = roi_width / static_cast<T>(pooled_width);

            int hstart = floor(
                static_cast<T>(ph) * bin_size_h + roi_start_h);
            int wstart = floor(
                static_cast<T>(pw) * bin_size_w + roi_start_w);
            int hend = ceil(
                static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
            int wend = ceil(
                static_cast<T>(pw + 1) * bin_size_w + roi_start_w);

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height);
            wstart = min(max(wstart, 0), width);
            hend = min(max(hend, 0), height);
            wend = min(max(wend, 0), width);

            // Compute c at bottom
            int gh = floor(
                static_cast<T>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<T>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int data_offset = (roi_batch_ind * channels + c) * height * width;
            T maxval = - (T) (1.0 / 0.0);
            int maxidx = -1;
            for (int h = hstart; h < hend; ++h){
              for (int w = wstart; w < wend; ++w){
                 int bottom_index = h * width + w;
                 float tmpval = bottom_data[data_offset + bottom_index];
                 if (tmpval > maxval) {
                     maxval = tmpval;
                     maxidx =  bottom_index;
                 }
              }
            }

            top_data = maxval;
            argmax_data = maxidx;
            ''', 'ps_roi_max_pooling_2d_fwd'
        )(bottom_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channel, height, width,
          out_c, self.out_h, self.out_w, self.group_size,
          top_data, self.argmax_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        top_diff = gy[0]
        height, width = self._bottom_data_shape[2:]
        bottom_diff = np.zeros(self._bottom_data_shape, np.float32)

        spatial_scale = self.spatial_scale
        pooled_height = self.out_h
        pooled_width = self.out_w
        group_size = self.group_size

        for i in six.moves.range(top_diff.size):
            n, ctop, ph, pw = np.unravel_index(i, top_diff.shape)

            roi_batch_ind = int(bottom_roi_indices[n])
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 0.1)
            roi_width = max(roi_end_w - roi_start_w, 0.1)
            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            hstart = int(np.floor(ph * bin_size_h + roi_start_h))
            wstart = int(np.floor(pw * bin_size_w + roi_start_w))
            hend = int(np.ceil((ph + 1) * bin_size_h + roi_start_h))
            wend = int(np.ceil((pw + 1) * bin_size_w + roi_start_w))
            hstart = min(max(hstart, 0), height)
            wstart = min(max(wstart, 0), width)
            hend = min(max(hend, 0), height)
            wend = min(max(wend, 0), width)

            gh = int(np.floor(ph * group_size / pooled_height))
            gw = int(np.floor(pw * group_size / pooled_width))
            gh = min(max(gh, 0), group_size - 1)
            gw = min(max(gw, 0), group_size - 1)
            c = (ctop * group_size + gh) * group_size + gw

            maxidx = self.argmax_data[n, ctop, ph, pw]
            if maxidx != -1:
                y = int(maxidx / width)
                x = maxidx % width
                bottom_diff[roi_batch_ind, c, y, x] \
                    += top_diff[n, ctop, ph, pw]

        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
        out_c, out_h, out_w = gy[0].shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, np.float32)
        cuda.elementwise(
            '''
            raw T top_diff, raw int32 argmax_data,
            raw T bottom_rois, raw int32 bottom_roi_indices,
            T spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size
            ''',
            'raw T bottom_diff',
            '''
            int ph = (i / pooled_width) % pooled_height;
            int pw = i % pooled_width;
            int ctop = (i / pooled_width / pooled_height) % pooled_dim;
            int n = i / pooled_width / pooled_height / pooled_dim;

            // [start, end) interval for spatial sampling
            int roi_batch_ind = bottom_roi_indices[n];
            T roi_start_h = bottom_rois[n * 4 + 0] * spatial_scale;
            T roi_start_w = bottom_rois[n * 4 + 1] * spatial_scale;
            T roi_end_h = bottom_rois[n * 4 + 2] * spatial_scale;
            T roi_end_w = bottom_rois[n * 4 + 3] * spatial_scale;

            // Force too small ROIs to be 1x1
            T roi_height = max(roi_end_h - roi_start_h, 0.1);
            T roi_width = max(roi_end_w - roi_start_w, 0.1); // avoid 0

            // Compute w and h at bottom
            T bin_size_h = roi_height / static_cast<T>(pooled_height);
            T bin_size_w = roi_width / static_cast<T>(pooled_width);

            int hstart = floor(
                static_cast<T>(ph) * bin_size_h + roi_start_h);
            int wstart = floor(
                static_cast<T>(pw) * bin_size_w + roi_start_w);
            int hend = ceil(
                static_cast<T>(ph + 1.0) * bin_size_h + roi_start_h);
            int wend = ceil(
                static_cast<T>(pw + 1.0) * bin_size_w + roi_start_w);

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart, 0), height);
            wstart = min(max(wstart, 0), width);
            hend = min(max(hend, 0), height);
            wend = min(max(wend, 0), width);

            // Compute c at bottom
            int gh = floor(
                static_cast<T>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<T>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int bottom_diff_offset = (roi_batch_ind * channels + c);
            bottom_diff_offset = bottom_diff_offset * height * width;
            int top_diff_offset =
                (n * pooled_dim + ctop) * pooled_height * pooled_width;

            int maxidx =
                argmax_data[top_diff_offset + ph * pooled_width + pw];
            if (maxidx != -1) {
                atomicAdd(
                    &bottom_diff[bottom_diff_offset + maxidx],
                    top_diff[top_diff_offset + ph * pooled_width + pw]);
            }
            ''', 'ps_roi_max_pooling_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          out_c, out_h, out_w, self.group_size, bottom_diff,
          size=gy[0].size)

        return bottom_diff, None, None


def ps_roi_max_pooling_2d(
        x, rois, roi_indices, outsize,
        spatial_scale, group_size
):
    """Position Sensitive Region of Interest (ROI) Max pooling function.

    This function computes position sensitive max of input spatial patch
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
        outsize ((int, int, int) or (int, int) or int): Expected output size
            after pooled: (channel, height, width) or (height, width)
            or outsize. ``outsize=o`` and ``outsize=(o, o)`` are equivalent.
            Channel parameter is used to assert the input shape.
        spatial_scale (float): Scale of the roi is resized.
        group_size (int): Position sensitive group size.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing PSROIPooling:
    `R-FCN <https://arxiv.org/abs/1605.06409>`_.

    """
    return PSROIMaxPooling2D(outsize, spatial_scale,
                             group_size)(x, rois, roi_indices)
