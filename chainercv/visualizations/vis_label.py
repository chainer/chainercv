from __future__ import division

import numpy as np


def vis_label(label, label_names=None, alpha=1, ax=None):
    """Visualize label of semantic segmentation.

    Example:

        >>> from chainercv.datasets import VOCSemanticSegmentationDataset
        >>> from chainercv.datasets \
        ...     import voc_semantic_segmentation_label_names
        >>> from chainercv.visualizations import vis_image
        >>> from chainercv.visualizations import vis_label
        >>> import matplotlib.pyplot as plot
        >>> dataset = VOCSemanticSegmentationDataset()
        >>> img, label = dataset[60]
        >>> ax = vis_image(img)
        >>> vis_label(
        ...     label, label_names=voc_semantic_segmentation_label_names,
        ...     alpha=0.75, ax=ax)
        >>> plot.show()

    Args:
        label (~numpy.ndarray): An integer array of shape
            :math:`(height, width)`.
            The values correspond to id for label names stored in
            :obj:`label_names`.
        label_names (iterable of strings): Name of labels ordered according
            to label ids.
        alpha (float): The value which determines transparency of the figure.
            The range of this value is :math:`[0, 1]`. If this
            value is :obj:`0`, the figure will be completely transparent.
            The default value is :obj:`1`. This option is useful for
            overlaying labels on the source images.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plot

    if label_names is None:
        n_class = label.max()
    else:
        n_class = len(label_names) + 1

    cmap = plot.get_cmap()

    img = cmap(label / (n_class - 1))
    # if label is invalid, alpha = 0
    # otherwise, alpha = alpha
    img[:, :, 3] = np.where(label >= 0, alpha, 0)

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.imshow(img)

    return ax
