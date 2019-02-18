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

import numpy as np
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check


def _pair(x):
    if isinstance(x, chainer.utils.collections_abc.Iterable):
        return x
    return x, x


def _get_bounds(p, limit):
    if p < -1 or p > limit:
        # out of range, so it is empty
        return None, None, None
    if p <= 0:
        p = 0
    low = int(np.floor(p))
    if low >= limit - 1:
        high = low = limit - 1
        p = float(low)
    else:
        high = low + 1
    return p, low, high


def _get_bilinear_interp_params(y, x, y_low, x_low, y_high, x_high):
    ly = y - y_low
    lx = x - x_low
    hy = 1. - ly
    hx = 1. - lx
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
    if (p <= 0) {
        p = 0;
    }
    low = (int)p;
    if (low >= limit - 1) {
        high = low = limit - 1;
        p = (T)low;
    } else {
        high = low + 1;
    }
    return true;
}

__device__
void get_bilinear_interp_params(
    T y, T x, int y_low, int x_low, int y_high, int x_high,
    T &w1, T &w2, T &w3, T &w4) {
    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly;
    T hx = 1. - lx;
    w1 = hy * hx;
    w2 = hy * lx;
    w3 = ly * hx;
    w4 = ly * lx;
}
'''


class PSROIAverageAlign2D(function.Function):

    def __init__(
            self, out_c, out_h, out_w, spatial_scale,
            group_size, sampling_ratio=None
    ):
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
        if not (isinstance(spatial_scale, float) and spatial_scale > 0):
            raise TypeError(
                'spatial_scale must be a positive float number: {}, {}'
                .format(type(spatial_scale), spatial_scale))
        sampling_ratio = _pair(sampling_ratio)
        if not all((isinstance(s, int) and s >= 1) or s is None
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
        channels, height, width = bottom_data.shape[1:]
        n_roi = bottom_rois.shape[0]
        top_data = np.empty(
            (n_roi, self.out_c, self.out_h, self.out_w), dtype=np.float32)

        group_size = self.group_size
        pooled_dim, pooled_width, pooled_height \
            = self.out_c, self.out_w, self.out_h
        spatial_scale = self.spatial_scale

        for i in six.moves.range(top_data.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            ctop = int(i / pooled_width / pooled_height) % pooled_dim
            n = int(i / pooled_width / pooled_height / pooled_dim)

            roi_batch_ind = int(bottom_roi_indices[n])
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 1.)
            roi_width = max(roi_end_w - roi_start_w, 1.)
            bin_size_h = 1. * roi_height / pooled_height
            bin_size_w = 1. * roi_width / pooled_width

            gh = np.floor(float(ph) * group_size / pooled_height)
            gw = np.floor(float(pw) * group_size / pooled_width)
            gh = int(min(max(gh, 0), group_size - 1))
            gw = int(min(max(gw, 0), group_size - 1))
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

                    v1 = bottom_data[roi_batch_ind, c, y_low, x_low]
                    v2 = bottom_data[roi_batch_ind, c, y_low, x_high]
                    v3 = bottom_data[roi_batch_ind, c, y_high, x_low]
                    v4 = bottom_data[roi_batch_ind, c, y_high, x_high]

                    output_val += w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

                    # }}

            output_val /= count
            top_data[n, ctop, ph, pw] = output_val

        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_roi = bottom_rois.shape[0]
        top_data = cuda.cupy.empty(
            (n_roi, self.out_c, self.out_h, self.out_w), dtype=np.float32)

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
            T spatial_scale, int32 channels, int32 height, int32 width,
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
            T roi_start_h = static_cast<T>(
                round(bottom_rois[n * 4 + 0])) * spatial_scale;
            T roi_start_w = static_cast<T>(
                round(bottom_rois[n * 4 + 1])) * spatial_scale;
            T roi_end_h = static_cast<T>(
                round(bottom_rois[n * 4 + 2])) * spatial_scale;
            T roi_end_w = static_cast<T>(
                round(bottom_rois[n * 4 + 3])) * spatial_scale;

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
                (roi_batch_ind * channels + c) * height * width;

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

                    T v1 = bottom_data[bottom_data_offset +
                                           y_low * width + x_low];
                    T v2 = bottom_data[bottom_data_offset +
                                           y_low * width + x_high];
                    T v3 = bottom_data[bottom_data_offset +
                                           y_high * width + x_low];
                    T v4 = bottom_data[bottom_data_offset +
                                           y_high * width + x_high];
                    // }}

                    output_val += (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
                }
            }
            output_val /= count;

            top_data = output_val;
            ''',
            'ps_roi_average_align_2d_fwd',
            preamble=_GET_BILINEAR_INTERP_KERNEL,
        )(bottom_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          self.out_c, self.out_h, self.out_w, self.group_size,
          sampling_ratio_h, sampling_ratio_w, top_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = np.zeros(self._bottom_data_shape, np.float32)

        spatial_scale = self.spatial_scale
        pooled_dim = self.out_c
        pooled_height = self.out_h
        pooled_width = self.out_w
        group_size = self.group_size
        top_diff = gy[0]

        for i in six.moves.range(top_diff.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            ctop = int(i / pooled_width / pooled_height) % pooled_dim
            n = int(i / pooled_width / pooled_height / pooled_dim)

            roi_batch_ind = int(bottom_roi_indices[n])
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_width = max(roi_end_w - roi_start_w, 1.)
            roi_height = max(roi_end_h - roi_start_h, 1.)
            bin_size_h = 1. * roi_height / pooled_height
            bin_size_w = 1. * roi_width / pooled_width

            gh = np.floor(float(ph) * group_size / pooled_height)
            gw = np.floor(float(pw) * group_size / pooled_width)
            gh = int(min(max(gh, 0), group_size - 1))
            gw = int(min(max(gw, 0), group_size - 1))
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

                    g1 = top_diff_this_bin * w1 / count
                    g2 = top_diff_this_bin * w2 / count
                    g3 = top_diff_this_bin * w3 / count
                    g4 = top_diff_this_bin * w4 / count

                    if (x_low >= 0 and x_high >= 0 and
                            y_low >= 0 and y_high >= 0):
                        bottom_diff[roi_batch_ind, c, y_low, x_low] += g1
                        bottom_diff[roi_batch_ind, c, y_low, x_high] += g2
                        bottom_diff[roi_batch_ind, c, y_high, x_low] += g3
                        bottom_diff[roi_batch_ind, c, y_high, x_high] += g4
                    # }}

        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
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
            T spatial_scale, int32 channels, int32 height, int32 width,
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
            T roi_start_h = static_cast<T>(
                round(bottom_rois[n * 4 + 0])) * spatial_scale;
            T roi_start_w = static_cast<T>(
                round(bottom_rois[n * 4 + 1])) * spatial_scale;
            T roi_end_h = static_cast<T>(
                round(bottom_rois[n * 4 + 2])) * spatial_scale;
            T roi_end_w = static_cast<T>(
                round(bottom_rois[n * 4 + 3])) * spatial_scale;

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
                (roi_batch_ind * channels + c) * height * width;

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

                    T g1 = top_diff_this_bin * w1 / count;
                    T g2 = top_diff_this_bin * w2 / count;
                    T g3 = top_diff_this_bin * w3 / count;
                    T g4 = top_diff_this_bin * w4 / count;

                    if (x_low >= 0 && x_high >= 0 &&
                            y_low >= 0 && y_high >= 0) {
                        atomicAdd(&bottom_diff[bottom_diff_offset +
                                               y_low * width + x_low], g1);
                        atomicAdd(&bottom_diff[bottom_diff_offset +
                                               y_low * width + x_high], g2);
                        atomicAdd(&bottom_diff[bottom_diff_offset +
                                               y_high * width + x_low], g3);
                        atomicAdd(&bottom_diff[bottom_diff_offset +
                                               y_high * width + x_high], g4);
                    }
                }
            }
            ''', 'ps_roi_average_align_2d_bwd',
            preamble=_GET_BILINEAR_INTERP_KERNEL,
        )(gy[0], bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          self.out_c, self.out_h, self.out_w,
          self.group_size, sampling_ratio_h, sampling_ratio_w,
          bottom_diff, size=gy[0].size)

        return bottom_diff, None, None


def ps_roi_average_align_2d(
        x, rois, roi_indices, out_c, out_h, out_w,
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
        out_c (int): Channels of output image after pooled.
        out_h (int): Height of output image after pooled.
        out_w (int): Width of output image after pooled.
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
        out_c, out_h, out_w, spatial_scale,
        group_size, sampling_ratio)(x, rois, roi_indices)
