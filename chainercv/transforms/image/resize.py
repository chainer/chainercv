from __future__ import division

import math
import numpy as np
import PIL
import warnings

import chainer


try:
    import cv2

    def _resize_cpu(img, size, interpolation):
        img = img.transpose(1, 2, 0)
        if interpolation == PIL.Image.NEAREST:
            cv_interpolation = cv2.INTER_NEAREST
        elif interpolation == PIL.Image.BILINEAR:
            cv_interpolation = cv2.INTER_LINEAR
        elif interpolation == PIL.Image.BICUBIC:
            cv_interpolation = cv2.INTER_CUBIC
        elif interpolation == PIL.Image.LANCZOS:
            cv_interpolation = cv2.INTER_LANCZOS4
        H, W = size
        img = cv2.resize(img, dsize=(W, H), interpolation=cv_interpolation)

        # If input is a grayscale image, cv2 returns a two-dimentional array.
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        return img.transpose(2, 0, 1)

except ImportError:
    def _resize_cpu(img, size, interpolation):
        warnings.warn(
            'cv2 is not installed on your environment. '
            'ChainerCV will fall back on Pillow. '
            'Installation of cv2 is recommended for faster computation. ',
            RuntimeWarning)

        C = img.shape[0]
        H, W = size
        out = np.empty((C, H, W), dtype=img.dtype)
        for ch, out_ch in zip(img, out):
            ch = PIL.Image.fromarray(ch, mode='F')
            out_ch[:] = ch.resize((W, H), resample=interpolation)
        return out


if chainer.cuda.available:
    import cupy as cp

    @cp.util.memoize(for_each_device=True)
    def _load_kernel(kernel_name, code, options=()):
        assert isinstance(options, tuple)
        kernel_code = cp.cuda.compile_with_cache(code, options=options)
        return kernel_code.get_function(kernel_name)


def resize(img, size, interpolation=PIL.Image.BILINEAR):
    """Resize image to match the given shape.

    In CPU mode, this method uses :mod:`cv2` or :mod:`PIL` for the backend.
    If :mod:`cv2` is installed, this function uses the implementation in
    :mod:`cv2`. This implementation is faster than the implementation in
    :mod:`PIL`. Under Anaconda environment,
    :mod:`cv2` can be installed by the following command.

    .. code::

        $ conda install -c menpo opencv3=3.2.0

    Args:
        img (array): An array to be transformed.
            This is in CHW format and the type should be :obj:`numpy.float32`.
        size (tuple): This is a tuple of length 2. Its elements are
            ordered as (height, width).
        interpolation (int): Determines sampling strategy. This is one of
            :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BILINEAR`,
            :obj:`PIL.Image.BICUBIC`, :obj:`PIL.Image.LANCZOS`.
            Bilinear interpolation is the default strategy.

    Returns:
        array: A resize array in CHW format.

    """
    xp = chainer.cuda.get_array_module(img)
    if xp == np:
        return _resize_cpu(img, size, interpolation)
    else:
        return _resize_gpu(img, size, interpolation)


resize_linear_code = '''
extern "C"
__global__ void resize_linear(
    const float* src, float* dst, const float fy, const float fx,
    const int src_row, const int src_col,
    const int dst_row, const int dst_col
)
{
    const int dst_z = blockIdx.z;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (dst_y < dst_row && dst_x < dst_col)
    {
        const float src_y = float(dst_y) * fy;
        const float src_x = float(dst_x) * fx;

        const int y1 = int(src_y);
        const int x1 = int(src_x);
        const int y2 = y1 + 1;
        const int x2 = x1 + 1;
        const int y2_read = min(y2, src_row - 1);
        const int x2_read = min(x2, src_col - 1);

        const float w1 = (y2 - src_y) * (x2 - src_x);
        const float w2 = (y2 - src_y) * (src_x - x1);
        const float w3 = (src_y - y1) * (x2 - src_x);
        const float w4 = (src_y - y1) * (src_x - x1);

        float out;
        float src_reg;
        int src_row_col = src_row * src_col;
        int dst_row_col = dst_row * dst_col;
        src_reg = src[dst_z * src_row_col + y1 * src_col + x1];
        out = src_reg * w1;

        src_reg = src[dst_z * src_row_col + y1 * src_col + x2_read];
        out = out + src_reg * w2;

        src_reg = src[dst_z * src_row_col + y2_read * src_col + x1];
        out = out + src_reg * w3;

        src_reg = src[dst_z * src_row_col + y2_read * src_col + x2_read];
        out = out + src_reg * w4;

        dst[dst_z * dst_row_col + dst_y * dst_col + dst_x] = out;
    }
}
'''


def _resize_gpu(src, size, interpolation=PIL.Image.BILINEAR):
    if interpolation != PIL.Image.BILINEAR:
        raise ValueError('GPU resize only supports bilinear interpolation.')
    if src.dtype != np.float32:
        raise ValueError('Only supports float32 images.')

    out_shape = (src.shape[0],) + size
    dst = cp.zeros(out_shape, dtype=np.float32)
    _, src_H, src_W = src.shape
    _, dst_H, dst_W = dst.shape

    fy = (src_H - 1) / (dst_H - 1)
    fx = (src_W - 1) / (dst_W - 1)

    kernel = _load_kernel('resize_linear', resize_linear_code)
    block = (32, 8, 1)
    grid = (math.ceil(dst_W / block[0]),
            math.ceil(dst_H / block[1]),
            src.shape[0])

    src = cp.ascontiguousarray(src)
    args = (src, dst, cp.float32(fy), cp.float32(fx),
            src_H, src_W, dst_H, dst_W)

    kernel(grid, block, args=args)
    return dst
