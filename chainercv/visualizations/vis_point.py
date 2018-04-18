from __future__ import division

import numpy as np
import six

from chainercv.visualizations.vis_image import vis_image


def vis_point(img, point, mask=None, ax=None):
    """Visualize points in an image.

    Example:

        >>> import chainercv
        >>> import matplotlib.pyplot as plt
        >>> dataset = chainercv.datasets.CUBPointDataset()
        >>> img, point, mask = dataset[0]
        >>> chainercv.visualizations.vis_point(img, point, mask)
        >>> plt.show()

    Args:
        img (~numpy.ndarray): An image of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`. This should be visualizable using
            :obj:`matplotlib.pyplot.imshow(img)`
        point (~numpy.ndarray): An array of point coordinates whose shape is
            :math:`(P, 2)`, where :math:`P` is
            the number of points.
            The second axis corresponds to :math:`y` and :math:`x` coordinates
            of the points.
        mask (~numpy.ndarray): A boolean array whose shape is
            :math:`(P,)`. If :math:`i` th element is :obj:`True`, the
            :math:`i` th point is not displayed. If not specified,
            all points in :obj:`point` will be displayed.
        ax (matplotlib.axes.Axes): If provided, plot on this axis.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    import matplotlib.pyplot as plt
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    _, H, W = img.shape
    n_point = len(point)

    if mask is None:
        mask = np.ones((n_point,), dtype=np.bool)

    cm = plt.get_cmap('gist_rainbow')

    colors = [cm(i / n_point) for i in six.moves.range(n_point)]

    for i in range(n_point):
        if mask[i]:
            ax.scatter(point[i][1], point[i][0], c=colors[i], s=100)

    ax.set_xlim(left=0, right=W)
    ax.set_ylim(bottom=H - 1, top=0)
    return ax
