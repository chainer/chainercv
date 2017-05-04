import numpy as np

from chainercv.visualizations.vis_image import vis_image


def vis_bbox(img, bbox, label=None, score=None, label_names=None, ax=None):
    """Visualize bounding boxes inside image.

    Example:

        >>> import chainercv
        >>> import matplotlib.pyplot as plot
        >>> dataset = chainercv.datasets.VOCDetectionDataset()
        >>> img, bbox, label = dataset[60]
        >>> chainercv.visualizations.vis_bbox(
                img, bbox, label, label_names=dataset.labels)
        >>> plot.show()

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in BGR format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :obj:`(x_min, y_min, x_max, y_max)` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label_ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plot
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    for i, bbox_elem in enumerate(bbox):
        xy = (bbox_elem[0], bbox_elem[1])
        width = bbox_elem[2] - bbox_elem[0]
        height = bbox_elem[3] - bbox_elem[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=3))

        caption = list()

        if label is not None and label_names is not None:
            label_elem = label[i]
            caption.append(label_names[label_elem])
        if score is not None:
            score_elem = score[i]
            caption.append('{:.2f}'.format(score_elem))

        if len(caption) > 0:
            ax.text(bbox_elem[0], bbox_elem[1],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax
