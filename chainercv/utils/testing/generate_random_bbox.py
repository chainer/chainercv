import numpy as np


def generate_random_bbox(n, img_size, min_length, max_length):
    """Generate valid bounding boxes with random position and shape.

    Args:
        n (int): The number of bounding boxes.
        img_size (tuple): A tuple of length 2. The height and the width
            of the image on which bounding boxes locate.
        min_length (float): The minimum length of edges of bounding boxes.
        max_length (float): The maximum length of edges of bounding boxes.

    Return:
        numpy.ndarray:
        Coordinates of bounding boxes. Its shape is :math:`(R, 4)`. \
        Here, :math:`R` equals :obj:`n`.
        The second axis contains :math:`y_{min}, x_{min}, y_{max}, x_{max}`,
        where
        :math:`min\_length \\leq y_{max} - y_{min} < max\_length`.
        and
        :math:`min\_length \\leq x_{max} - x_{min} < max\_length`

    """
    H, W = img_size
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((y_min, x_min, y_max, x_max), axis=1).astype(np.float32)
    return bbox
