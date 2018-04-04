from __future__ import division

import numpy as np

from chainercv.utils.mask.mask_to_bbox import mask_to_bbox
from chainercv.visualizations.colormap import voc_colormap


def vis_instance_segmentation(
        img, mask, label=None, score=None, label_names=None,
        colors=None, alpha=0.7, ax=None):
    """Visualize instance segmentation.

    Example:

        >>> from chainercv.datasets import SBDInstanceSegmentationDataset
        >>> from chainercv.datasets \
        ...     import sbd_instance_segmentation_label_names
        >>> from chainercv.visualizations import vis_instance_segmentation
        >>> import matplotlib.pyplot as plot
        >>> dataset = SBDInstanceSegmentationDataset()
        >>> img, mask, label = dataset[0]
        >>> vis_instance_segmentation(
        ...     img, mask, label,
        ...     label_names=sbd_instance_segmentation_label_names)
        >>> plot.show()

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, H, W)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        mask (~numpy.ndarray): A bool array of shape
            :math`(R, H, W)`.
            If there is an object, the value of the pixel is :obj:`True`,
            and otherwise, it is :obj:`False`.
        label (~numpy.ndarray): An integer array of shape :math:`(R, )`.
            The values correspond to id for label names stored in
            :obj:`label_names`.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids.
        colors: (iterable of tuple): List of colors.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`. The :obj:`i`-th element is the color used
            to visualize the :obj:`i`-th instance.
            If :obj:`colors` is :obj:`None`, the default color map is used.
        alpha (float): The value which determines transparency of the figure.
            The range of this value is :math:`[0, 1]`. If this
            value is :obj:`0`, the figure will be completely transparent.
            The default value is :obj:`0.7`. This option is useful for
            overlaying the label on the source image.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        matploblib.axes.Axes: Returns :obj:`ax`.
        :obj:`ax` is an :class:`matploblib.axes.Axes` with the plot.

    """
    from matplotlib import pyplot as plot
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)

    bbox = mask_to_bbox(mask)

    if len(bbox) != len(mask):
        raise ValueError('The length of mask must be same as that of bbox')
    if label is not None and len(bbox) != len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and len(bbox) != len(score):
        raise ValueError('The length of score must be same as that of bbox')

    n_inst = len(bbox)
    if colors is None:
        colors = [voc_colormap(l) for l in range(1, n_inst + 1)]
    colors = np.array(colors)

    canvas_img = img.transpose((1, 2, 0)).copy()
    for i, (bb, msk) in enumerate(zip(bbox, mask)):
        # The length of `colors` can be smaller than the number of instances
        # if a non-default `colors` is used.
        color = colors[i % len(colors)]
        bb = np.round(bb).astype(np.int32)
        y_min, x_min, y_max, x_max = bb
        if y_max > y_min and x_max > x_min:
            canvas_img[msk] = alpha * color + canvas_img[msk] * (1 - alpha)

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
            ax.text((x_max + x_min) / 2, y_min,
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': color / 255, 'alpha': alpha},
                    fontsize=8, color='white')

    ax.imshow(canvas_img.astype(np.uint8))
    return ax
