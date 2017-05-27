from __future__ import division

import numpy as np


def vis_label(label, label_names=None, alpha=1, ax=None):
    """Visualize a label for semantic segmentation.

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
        >>> _, legned_handles = vis_label(
        ...     label, label_names=voc_semantic_segmentation_label_names,
        ...     alpha=0.9, ax=ax)
        >>> ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc=2)
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
            overlaying the label on the source image.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        matploblib.axes.Axes and list of matplotlib.patches.Patch:
        Returns :obj:`ax` and :obj:`legend_handles`.
        :obj:`ax` is an :class:`matploblib.axes.Axes` with the plot.
        It can be used for further tweaking.
        :obj:`legend_handles` is a list of legends. It can be passed
        :func:`matploblib.pyplot.legend` to show a legend.

    """
    from matplotlib.patches import Patch
    from matplotlib import pyplot as plot

    if label_names is None:
        label_names = [str(l) for l in range(label.max() + 1)]
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

    legend_handles = list()
    for l, label_name in enumerate(label_names):
        legend_handles.append(
            Patch(color=cmap(l / (n_class - 1)), label=label_name))

    return ax, legend_handles
