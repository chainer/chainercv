from __future__ import division

import numpy as np
import six

from chainercv.visualizations.vis_image import vis_image


def vis_point(img, point, visible=None, ax=None):
    """Visualize points in an image.

    Example:

        >>> import chainercv
        >>> import matplotlib.pyplot as plt
        >>> dataset = chainercv.datasets.CUBKeypointDataset()
        >>> img, point, visible = dataset[0]
        >>> chainercv.visualizations.vis_point(img, point, visible)
        >>> plt.show()

    Args:
        img (~numpy.ndarray): See the table below.
            If this is :obj:`None`, no image is displayed.
        point (~numpy.ndarray): See the table below.
        visible (~numpy.ndarray): See the table below.
        ax (matplotlib.axes.Axes): If provided, plot on this axis.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`point`, ":math:`[(K, 2)]` or :math:`(R, K, 2)`", \
        :obj:`float32`, ":math:`(y, x)`"
        :obj:`visible`, ":math:`[(K,)]` or :math:`(R, K)`", :obj:`bool`, --

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    import matplotlib.pyplot as plt
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    _, H, W = img.shape
    n_inst = len(point)

    cm = plt.get_cmap('gist_rainbow')

    for i in range(n_inst):
        pnt = point[i]
        n_point = len(pnt)
        if visible is not None:
            vsble = visible[i]
        else:
            vsble = np.ones((n_point,), dtype=np.bool)

        colors = [cm(k / n_point) for k in six.moves.range(n_point)]

        for k in range(n_point):
            if vsble[k]:
                ax.scatter(pnt[k][1], pnt[k][0], c=colors[k], s=100)

    ax.set_xlim(left=0, right=W)
    ax.set_ylim(bottom=H - 1, top=0)
    return ax
