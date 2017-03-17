import numpy as np
import six


def vis_keypoint(img, keypoint, kp_mask=None, ax=None):
    """Visualize keypoints in an image.

    Example:

        >>> import chainercv
        >>> import matplotlib.pyplot as plot
        >>> dataset = chainercv.datasets.CUBKeypointDataset()
        >>> img, keypoint, kp_mask = dataset[0]
        >>> img = chainercv.transforms.chw_to_pil_image(img)
        >>> chainercv.tasks.vis_keypoint(img, keypoint, kp_mask)
        >>> plot.show()

    Args:
        img (~numpy.ndarray): an image of shape
            :math:`(height, width, 3)`. This should be visualizable using
            :obj:`matplotlib.pyplot.imshow(img)`
        keypoint (~numpy.ndarray): An array with keypoint pairs whose shape is
            :math:`(K, 2)`, where :math:`K` is
            the number of keypoints in the array.
            The second axis corresponds to :math:`x` and :math:`y` coordinates
            of the keypoint.
        kp_mask (~numpy.ndarray, optional): A boolean array whose shape is
            :math:`(K,)`. If :math:`i` th index is :obj:`True`, the
            :math:`i` th keypoint is not displayed. If not specified,
            all keypoints in :obj:`keypoint` will be displayed.
        ax (matplotlib.axes.Axes, optional): If provided, plot on this axis.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    import matplotlib.pyplot as plot

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)

    H, W, _ = img.shape
    n_kp = len(keypoint)

    if kp_mask is None:
        kp_mask = np.ones((n_kp,), dtype=np.bool)

    cm = plot.get_cmap('gist_rainbow')

    colors = [cm(1. * i / n_kp) for i in six.moves.range(n_kp)]

    ax.imshow(img)
    for i in range(n_kp):
        if kp_mask[i]:
            ax.scatter(keypoint[i][0], keypoint[i][1], c=colors[i], s=100)

    ax.set_xlim(left=0, right=W)
    ax.set_ylim(bottom=H - 1, top=0)
    return ax
