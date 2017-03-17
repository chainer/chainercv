import numpy as np


def vis_bbox(img, bbox, label=None, label_names=None, ax=None):
    """Visualize bounding boxes inside image.

    Example:

        >>> import chainercv
        >>> import matplotlib.pyplot as plot
        >>> dataset = chainercv.datasets.VOCDetectionDataset()
        >>> img, bbox, label = dataset[60]
        >>> img = chainercv.transforms.chw_to_pil_image(img)
        >>> chainercv.tasks.vis_bbox(img, bbox, label, dataset.labels)
        >>> plot.show()

    Args:
        img (~numpy.ndarray): array of shape :math:`(height, width, 3)`.
            This is in RGB format.
        bbox (~numpy.ndarray): an array of shape :math:`(R, 5)`, where
            :math:`R` is the number of bounding boxes in the image. Elements
            are organized
            by :obj:`(x_min, y_min, x_max, y_max)` in the second axis.
        label (~numpy.ndarray): an integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        label_names (iterable of strings): name of labels ordered according
            to label_ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plot
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

    for i, bbox_elem in enumerate(bbox):
        xy = (bbox_elem[0], bbox_elem[1])
        width = bbox_elem[2] - bbox_elem[0]
        height = bbox_elem[3] - bbox_elem[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=3))
        if label_names is not None:
            label_elem = label[i]
            ax.text(bbox_elem[0], bbox_elem[1],
                    label_names[label_elem.astype(np.int)],
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax
