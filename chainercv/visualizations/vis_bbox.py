import numpy as np

from chainercv.visualizations.vis_image import vis_image


def vis_bbox(img, bbox, label=None, score=None, label_names=None,
             instance_colors=None, alpha=1., linewidth=3.,
             sort_by_score=True, ax=None):
    """Visualize bounding boxes inside image.

    Example:

        >>> from chainercv.datasets import VOCBboxDataset
        >>> from chainercv.datasets import voc_bbox_label_names
        >>> from chainercv.visualizations import vis_bbox
        >>> import matplotlib.pyplot as plt
        >>> dataset = VOCBboxDataset()
        >>> img, bbox, label = dataset[60]
        >>> vis_bbox(img, bbox, label,
        ...          label_names=voc_bbox_label_names)
        >>> plt.show()

        This example visualizes by displaying the same colors for bounding
        boxes assigned to the same labels.

        >>> from chainercv.datasets import VOCBboxDataset
        >>> from chainercv.datasets import voc_bbox_label_names
        >>> from chainercv.visualizations import vis_bbox
        >>> from chainercv.visualizations.colormap import voc_colormap
        >>> import matplotlib.pyplot as plt
        >>> dataset = VOCBboxDataset()
        >>> img, bbox, label = dataset[61]
        >>> colors = voc_colormap(label + 1)
        >>> vis_bbox(img, bbox, label,
        ...          label_names=voc_bbox_label_names,
        ...          instance_colors=colors)
        >>> plt.show()

    Args:
        img (~numpy.ndarray): See the table below. If this is :obj:`None`,
            no image is displayed.
        bbox (~numpy.ndarray): See the table below.
        label (~numpy.ndarray): See the table below. This is optional.
        score (~numpy.ndarray): See the table below. This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        instance_colors (iterable of tuples): List of colors.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`. The :obj:`i`-th element is the color used
            to visualize the :obj:`i`-th instance.
            If :obj:`instance_colors` is :obj:`None`, the red is used for
            all boxes.
        alpha (float): The value which determines transparency of the
            bounding boxes. The range of this value is :math:`[0, 1]`.
        linewidth (float): The thickness of the edges of the bounding boxes.
        sort_by_score (bool): When :obj:`True`, instances with high scores
            are always visualized in front of instances with low scores.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label`, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`score`, ":math:`(R,)`", :obj:`float32`, --

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    from matplotlib import pyplot as plt

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    if sort_by_score and score is not None:
        order = np.argsort(score)
        bbox = bbox[order]
        score = score[order]
        if label is not None:
            label = label[order]
        if instance_colors is not None:
            instance_colors = np.array(instance_colors)[order]

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 255
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        caption = []

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
    return ax
