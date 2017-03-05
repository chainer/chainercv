import numpy as np

try:
    import skimage.transform
    _available = True

except ImportError:
    _available = False


def resize(x, output_shape):
    """Resize image to match the given shape.

    Args:
        x (~numpy.ndarray): array to be transformed. This is in CHW format.
        output_shape (tuple): this is a tuple of length 2. Its elements are
            ordered as (height, width).

    """
    if not _available:
        raise ValueError('scikit-image is not installed on your environment, '
                         'so a function resize can not be '
                         ' used. Please install scikit-image.\n\n'
                         '  $ pip install scikit-image\n')

    if len(output_shape) != 2:
        raise ValueError('length of the output_shape needs to be 2')
    x = x.transpose(1, 2, 0)
    scale = np.max(np.abs(x))
    x = skimage.transform.resize(
        x / scale, output_shape).astype(x.dtype)
    x = x.transpose(2, 0, 1) * scale
    return x


if __name__ == '__main__':
    from skimage.data import astronaut
    data = astronaut().astype(np.float32)
    data = data.transpose(2, 0, 1)
    out = resize(data, (256, 128))
    import matplotlib.pyplot as plt
    plt.imshow(out.transpose(1, 2, 0).astype(np.uint8))
    plt.show()
