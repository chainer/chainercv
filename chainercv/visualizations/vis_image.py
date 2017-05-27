import numpy as np


def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plot
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose(1, 2, 0)
    ax.imshow(img.astype(np.uint8))
    return ax
