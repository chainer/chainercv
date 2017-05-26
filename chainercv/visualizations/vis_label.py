from chainercv.visualizations.vis_image import vis_image


def vis_label(img, label, alpha=0.5, label_names=None, ax=None):
    """Visualize label of semantic segmentation.

    Example:

        >>> from chainercv.datasets import VOCSemanticSegmentationDataset
        >>> from chainercv.datasets \
                import voc_semantic_segmentation_label_names
        >>> from chainercv.visualizations import vis_label
        >>> import matplotlib.pyplot as plot
        >>> dataset = VOCSemanticSegmentationDataset()
        >>> img, label = dataset[60]
        >>> vis_label(img, label,
        ...         label_names=voc_semantic segmentation_label_names)
        >>> plot.show()

    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`. If this argument is specified, this function
            blends :obj:`label` and :obj:`img` using :obj:`alpha`.
            If :obj:`img` is :obj:`None`, this function shows only :obj:`label`
            and ignores :obj:`alpha`.
        label (~numpy.ndarray): An integer array of shape
            :math:`(1, height, width)`.
            The values correspond to id for label names stored in
            :obj:`label_names`.
        alpha (float): The value which is used for blending :obj:`label` and
            :obj:`img`. The range of this value is :math:`[0, 1]`. If this
            value is :obj:`1`, this function shows only :obj:`label`.
            The default value is :obj:`0.5`.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
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

    if img is None:
        alpha = 1
    else:
        if not img.shape[1:] == label.shape[1:]:
            raise ValueError('The size of image must be same as that of label')
        vis_image(img, ax=ax)

    ax.imshow(label[0], vmin=-1, vmax=label.max(), alpha=alpha)

    return ax
