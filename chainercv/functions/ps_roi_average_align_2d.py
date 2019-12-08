# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2019 Preferred Infrastructure, Inc.
# Copyright (c) 2019 Preferred Networks, Inc.
# -----------------------------------------------------------------------------

# Original work:
# -----------------------------------------------------------------------------
# Copyright (c) 2015 by Contributors
# \file roi_pooling.cu
# \brief roi pooling operator
# \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
# \changed to roi_align by Elaine Bao
# \file roi_align.cu
# \roi align operator described in Mask RCNN
# -----------------------------------------------------------------------------

from __future__ import division

import numbers
import numpy as np
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check

from chainercv.functions.ps_roi_average_pooling_2d import _outsize


def _pair(x):
    if isinstance(x, chainer.utils.collections_abc.Iterable):
        return x
    return x, x


def _get_bounds(p, limit):
    if p < -1 or p > limit:
        # out of range, so it is empty
        return None, None, None

    low = int(np.floor(p))
    if low == limit:
        low = low - 1
    high = low + 1

    if low <= -1:
        p = 0
    elif high >= limit:
        p = limit - 1
    return p, low, high


def _get_bilinear_interp_params(y, x, y_low, x_low, y_high, x_high):
    ly = y - y_low
    lx = x - x_low
    hy = y_high - y
    hx = x_high - x
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx
    return w1, w2, w3, w4


_GET_BILINEAR_INTERP_KERNEL = '''
__device__
bool get_bounds(
    T &p, const int limit, int &low, int &high) {
    if (p < -1. || p > limit) {
        // empty
        return false;
    }
    low = (int)floor(p);
    if (low == limit) {
        low = low - 1;
    }
    high = low + 1;
    if (low <= -1) {
        p = (T) 0.0;
    } else if (high >= limit) {
        p = (T) (limit - 1);
    }
    return true;
}

__device__
void get_bilinear_interp_params(
    T y, T x, int y_low, int x_low, int y_high, int x_high,
    T &w1, T &w2, T &w3, T &w4) {
    T ly = y - y_low;
    T lx = x - x_low;
    T hy = y_high - y;
    T hx = x_high - x;
    w1 = hy * hx;
    w2 = hy * lx;
    w3 = ly * hx;
    w4 = ly * lx;
}
'''


class PSROIAverageAlign2D(function.Function):

    def __init__(
            self, outsize, spatial_scale,
            group_size, sampling_ratio=None
    ):
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
        if not (isinstance(group_size, numbers.Integral)
                and group_size > 0):
            raise TypeError(
                'group_size must be positive integer: {}, {}'
                .format(type(group_size), group_size))
        sampling_ratio = _pair(sampling_ratio)
        if not all((isinstance(s, numbers.Integral) and s >= 1) or s is None
                   for s in sampling_ratio):
            raise TypeError(
                'sampling_ratio must be integer >= 1 or a pair of it: {}'
                .format(sampling_ratio))

        self.out_c, self.out_h, self.out_w = out_c, out_h, out_w
        self.spatial_scale = spatial_scale
        self.group_size = group_size
        self.sampling_ratio = sampling_ratio

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

            gh = int(np.floor(ph * group_size / pooled_height))
            gw = int(np.floor(pw * group_size / pooled_width))
            gh = min(max(gh, 0), group_size - 1)
            gw = min(max(gw, 0), group_size - 1)
            c = (ctop * group_size + gh) * group_size + gw

            if self.sampling_ratio[0] is None:
                roi_bin_grid_h = int(np.ceil(roi_height / pooled_height))
            else:
                roi_bin_grid_h = self.sampling_ratio[0]
            if self.sampling_ratio[1] is None:
                roi_bin_grid_w = int(np.ceil(roi_width / pooled_width))
            else:
                roi_bin_grid_w = self.sampling_ratio[1]

            count = roi_bin_grid_h * roi_bin_grid_w

            output_val = 0.
            for iy in six.moves.range(roi_bin_grid_h):
                y = roi_start_h + ph * bin_size_h + \
                    (iy + .5) * bin_size_h / roi_bin_grid_h
                y, y_low, y_high = _get_bounds(y, height)
                if y is None or y_low is None or y_high is None:
                    continue
                for ix in six.moves.range(roi_bin_grid_w):
                    x = roi_start_w + pw * bin_size_w + \
                        (ix + .5) * bin_size_w / roi_bin_grid_w

                    x, x_low, x_high = _get_bounds(x, width)
                    if x is None or x_low is None or x_high is None:
                        continue

                    # bilinear interpolation {{
                    w1, w2, w3, w4 = _get_bilinear_interp_params(
                        y, x, y_low, x_low, y_high, x_high)

                    if w1 > 0 and y_low >= 0 and x_low >= 0:
                        v1 = bottom_data[roi_batch_ind, c, y_low, x_low]
                        output_val += w1 * v1

                    if w2 > 0 and y_low >= 0 and x_high <= width - 1:
                        v2 = bottom_data[roi_batch_ind, c, y_low, x_high]
                        output_val += w2 * v2

                    if w3 > 0 and y_high <= height - 1 and x_low >= 0:
                        v3 = bottom_data[roi_batch_ind, c, y_high, x_low]
                        output_val += w3 * v3

                    if w4 > 0 and y_high <= height - 1 and x_high <= width - 1:
                        v4 = bottom_data[roi_batch_ind, c, y_high, x_high]
                        output_val += w4 * v4

                    # }}

            output_val /= count
            top_data[n, ctop, ph, pw] = output_val

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

        if self.sampling_ratio[0] is None:
            sampling_ratio_h = 0
        else:
            sampling_ratio_h = self.sampling_ratio[0]
        if self.sampling_ratio[1] is None:
            sampling_ratio_w = 0
        else:
            sampling_ratio_w = self.sampling_ratio[1]
        cuda.elementwise(
            '''
            raw T bottom_data, raw T bottom_rois, raw int32 bottom_roi_indices,
            T spatial_scale, int32 channel, int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size, int32 sampling_ratio_h, int32 sampling_ratio_w
            ''',
            'T top_data',
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

            // Compute c at bottom
            int gh = floor(
                static_cast<T>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<T>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int bottom_data_offset =
                (roi_batch_ind * channel + c) * height * width;

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio_h > 0)
                ? sampling_ratio_h
                : ceil(roi_height / pooled_height);  // e.g. = 2
            int roi_bin_grid_w = (sampling_ratio_w > 0)
                ? sampling_ratio_w
                : ceil(roi_width / pooled_width);

            T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

            T output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g. iy = 0, 1
            {
                T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g. 0.5, 1.5
                int y_low, y_high;
                bool y_ret = get_bounds(y, height, y_low, y_high);
                if (!y_ret) continue;
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    T x = roi_start_w + pw * bin_size_w +
                        static_cast<T>(ix + .5f) * bin_size_w /
                            static_cast<T>(roi_bin_grid_w);

                    int x_low, x_high;
                    bool x_ret = get_bounds(x, width, x_low, x_high);
                    if (!x_ret) continue;
                    // bilinear_interpolation_gradient {{
                    T w1, w2, w3, w4;
                    get_bilinear_interp_params(
                        y, x, y_low, x_low, y_high, x_high, w1, w2, w3, w4);

                    if (w1 > 0 && y_low >= 0 && x_low >= 0) {
                        T v1 = bottom_data[
                            bottom_data_offset + y_low * width + x_low];
                        output_val += w1 * v1;
                    }
                    if (w2 > 0 && y_low >= 0 && x_high <= width - 1) {
                        T v2 = bottom_data[
                            bottom_data_offset + y_low * width + x_high];
                        output_val += w2 * v2;
                    }
                    if (w3 > 0 && y_high <= height - 1 && x_low >= 0) {
                        T v3 = bottom_data[
                            bottom_data_offset + y_high * width + x_low];
                        output_val += w3 * v3;
                    }
                    if (w4 > 0 && y_high <= height - 1 &&
                            x_high <= width - 1) {
                        T v4 = bottom_data[
                            bottom_data_offset + y_high * width + x_high];
                        output_val += w4 * v4;
                    }

                    // }}

                }
            }
            output_val /= count;

            top_data = output_val;
            ''',
            'ps_roi_average_align_2d_fwd',
            preamble=_GET_BILINEAR_INTERP_KERNEL,
        )(bottom_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channel, height, width,
          out_c, self.out_h, self.out_w, self.group_size,
          sampling_ratio_h, sampling_ratio_w, top_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        height, width = self._bottom_data_shape[2:]
        bottom_diff = np.zeros(self._bottom_data_shape, np.float32)

        spatial_scale = self.spatial_scale
        pooled_height = self.out_h
        pooled_width = self.out_w
        group_size = self.group_size
        top_diff = gy[0]

        for i in six.moves.range(top_diff.size):
            n, ctop, ph, pw = np.unravel_index(i, top_diff.shape)

            roi_batch_ind = int(bottom_roi_indices[n])
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_width = max(roi_end_w - roi_start_w, 0.1)
            roi_height = max(roi_end_h - roi_start_h, 0.1)
            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

            gh = int(np.floor(ph * group_size / pooled_height))
            gw = int(np.floor(pw * group_size / pooled_width))
            gh = min(max(gh, 0), group_size - 1)
            gw = min(max(gw, 0), group_size - 1)
            c = (ctop * group_size + gh) * group_size + gw

            top_diff_this_bin = top_diff[n, ctop, ph, pw]

            if self.sampling_ratio[0] is None:
                roi_bin_grid_h = int(np.ceil(roi_height / pooled_height))
            else:
                roi_bin_grid_h = self.sampling_ratio[0]
            if self.sampling_ratio[1] is None:
                roi_bin_grid_w = int(np.ceil(roi_width / pooled_width))
            else:
                roi_bin_grid_w = self.sampling_ratio[1]

            count = roi_bin_grid_h * roi_bin_grid_w

            for iy in six.moves.range(roi_bin_grid_h):
                y = roi_start_h + ph * bin_size_h + \
                    (iy + .5) * bin_size_h / roi_bin_grid_h
                y, y_low, y_high = _get_bounds(y, height)
                if y is None or y_low is None or y_high is None:
                    continue
                for ix in six.moves.range(roi_bin_grid_w):
                    x = roi_start_w + pw * bin_size_w + \
                        (ix + .5) * bin_size_w / roi_bin_grid_w

                    x, x_low, x_high = _get_bounds(x, width)
                    if x is None or x_low is None or x_high is None:
                        continue
                    # bilinear_interpolation_gradient {{
                    w1, w2, w3, w4 = _get_bilinear_interp_params(
                        y, x, y_low, x_low, y_high, x_high)

                    if w1 > 0 and y_low >= 0 and x_low >= 0:
                        g1 = top_diff_this_bin * w1 / count
                        bottom_diff[roi_batch_ind, c, y_low, x_low] += g1

                    if w2 > 0 and y_low >= 0 and x_high <= width - 1:
                        g2 = top_diff_this_bin * w2 / count
                        bottom_diff[roi_batch_ind, c, y_low, x_high] += g2

                    if w3 > 0 and y_high <= height - 1 and x_low >= 0:
                        g3 = top_diff_this_bin * w3 / count
                        bottom_diff[roi_batch_ind, c, y_high, x_low] += g3

                    if w4 > 0 and y_high <= height - 1 and x_high <= width - 1:
                        g4 = top_diff_this_bin * w4 / count
                        bottom_diff[roi_batch_ind, c, y_high, x_high] += g4

                    # }}

        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channel, height, width = self._bottom_data_shape[1:]
        out_c, out_h, out_w = gy[0].shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, np.float32)

        if self.sampling_ratio[0] is None:
            sampling_ratio_h = 0
        else:
            sampling_ratio_h = self.sampling_ratio[0]
        if self.sampling_ratio[1] is None:
            sampling_ratio_w = 0
        else:
            sampling_ratio_w = self.sampling_ratio[1]
        cuda.elementwise(
            '''
            raw T top_diff, raw T bottom_rois, raw int32 bottom_roi_indices,
            T spatial_scale, int32 channel, int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size, int32 sampling_ratio_h, int32 sampling_ratio_w
            ''',
            'raw T bottom_diff',
            '''
            // (n, c, h, w) coords in bottom data
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int ctop = (i / pooled_width / pooled_height) % pooled_dim;
            int n = i / pooled_width / pooled_height / pooled_dim;

            // Do not using rounding; this implementation detail is critical
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

            // Compute c at bottom
            int gh = floor(
                static_cast<T>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<T>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int bottom_diff_offset =
                (roi_batch_ind * channel + c) * height * width;

            int top_offset =
                (n * pooled_dim + ctop) * pooled_height * pooled_width;
            T top_diff_this_bin =
                top_diff[top_offset + ph * pooled_width + pw];

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio_h > 0)
                ? sampling_ratio_h
                : ceil(roi_height / pooled_height); // e.g. = 2
            int roi_bin_grid_w = (sampling_ratio_w > 0)
                ? sampling_ratio_w
                : ceil(roi_width / pooled_width);

            // We do average (integral) pooling inside a bin
            T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                T y = roi_start_h + ph * bin_size_h +
                    static_cast<T>(iy + .5f) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);  // e.g. 0.5, 1.5
                int y_low, y_high;
                bool y_ret = get_bounds(y, height, y_low, y_high);
                if (!y_ret) continue;
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    T x = roi_start_w + pw * bin_size_w +
                        static_cast<T>(ix + .5f) * bin_size_w /
                            static_cast<T>(roi_bin_grid_w);

                    int x_low, x_high;
                    bool x_ret = get_bounds(x, width, x_low, x_high);
                    if (!x_ret) continue;
                    // bilinear_interpolation_gradient {{
                    T w1, w2, w3, w4;
                    get_bilinear_interp_params(
                        y, x, y_low, x_low, y_high, x_high, w1, w2, w3, w4);

                    if (w1 > 0 && y_low >= 0 && x_low >= 0) {
                        T g1 = top_diff_this_bin * w1 / count;
                        atomicAdd(&bottom_diff[
                            bottom_diff_offset + y_low * width + x_low], g1);
                    }
                    if (w2 > 0 && y_low >= 0 && x_high <= width - 1) {
                        T g2 = top_diff_this_bin * w2 / count;
                        atomicAdd(&bottom_diff[
                            bottom_diff_offset + y_low * width + x_high], g2);
                    }
                    if (w3 > 0 && y_high <= height - 1 && x_low >= 0) {
                        T g3 = top_diff_this_bin * w3 / count;
                        atomicAdd(&bottom_diff[
                            bottom_diff_offset + y_high * width + x_low], g3);
                    }
                    if (w4 > 0 && y_high <= height - 1 &&
                            x_high <= width - 1) {
                        T g4 = top_diff_this_bin * w4 / count;
                        atomicAdd(&bottom_diff[
                            bottom_diff_offset + y_high * width + x_high], g4);
                    }
                }
            }
            ''', 'ps_roi_average_align_2d_bwd',
            preamble=_GET_BILINEAR_INTERP_KERNEL,
        )(gy[0], bottom_rois, bottom_roi_indices,
          self.spatial_scale, channel, height, width,
          out_c, out_h, out_w, self.group_size,
          sampling_ratio_h, sampling_ratio_w, bottom_diff,
          size=gy[0].size)

        return bottom_diff, None, None


def ps_roi_average_align_2d(
        x, rois, roi_indices, outsize,
        spatial_scale, group_size, sampling_ratio=None
):
    """Position Sensitive Region of Interest (ROI) Average align function.

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
        outsize ((int, int, int) or (int, int) or int): Expected output size
            after pooled: (channel, height, width) or (height, width)
            or outsize. ``outsize=o`` and ``outsize=(o, o)`` are equivalent.
            Channel parameter is used to assert the input shape.
        spatial_scale (float): Scale of the roi is resized.
        group_size (int): Position sensitive group size.
        sampling_ratio ((int, int) or int): Sampling step for the alignment.
            It must be an integer over :math:`1` or :obj:`None`, and the value
            is automatically decided when :obj:`None` is passed.  Use of
            different ratio in height and width axis is also supported by
            passing tuple of int as ``(sampling_ratio_h, sampling_ratio_w)``.
            ``sampling_ratio=s`` and ``sampling_ratio=(s, s)`` are equivalent.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing PSROIPooling:
    `R-FCN <https://arxiv.org/abs/1605.06409>`_.
    See the original paper proposing ROIAlign:
    `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_.

    """
    return PSROIAverageAlign2D(
        outsize, spatial_scale,
        group_size, sampling_ratio)(x, rois, roi_indices)
