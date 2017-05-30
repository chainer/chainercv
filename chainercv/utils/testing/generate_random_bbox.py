import numpy as np


def generate_random_bbox(n, img_size, min_length, max_length):
    """Generate valid bounding boxes with random position and shape.

    Args:
        n (int): The number of bounding boxes.
        img_size (tuple): A tuple of length 2. The width and the height
            of the image on which bounding boxes locate.
        min_length (int): The minimum length of edges of bounding boxes.
        max_length (int): The maximum length of edges of bounding boxes.

    Return:
        numpy.ndarray:
        Coordinates of bounding boxes. Its shape is :math:`(R, 4)`. \
        Here, :math:`R` equals :obj:`n`.
        The second axis contains :obj:`x_min, y_min, x_max, y_max`,
        where :obj:`x_min + min_length <= x_max < x_min + max_length` and
        :obj:`y_min + min_length <= y_max < y_min + max_length`.

    """
    W, H = img_size
    x_min = np.random.uniform(0, W - max_length, size=(n,))
    y_min = np.random.uniform(0, H - max_length, size=(n,))
    x_max = x_min + np.random.uniform(min_length, max_length, size=(n,))
    y_max = y_min + np.random.uniform(min_length, max_length, size=(n,))
    bbox = np.stack((x_min, y_min, x_max, y_max), axis=1).astype(np.float32)
    return bbox
