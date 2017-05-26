def vis_label(label, alpha=1, label_names=None, ax=None):
    """Visualize label of semantic segmentation.

    Example:

        >>> from chainercv.datasets import VOCSemanticSegmentationDataset
        >>> from chainercv.datasets \
                import voc_semantic_segmentation_label_names
        >>> from chainercv.visualizations import vis_label
        >>> import matplotlib.pyplot as plot
        >>> dataset = VOCSemanticSegmentationDataset()
        >>> _, label = dataset[60]
        >>> vis_label(label,
        ...         label_names=voc_semantic segmentation_label_names)
        >>> plot.show()

    Args:
        label (~numpy.ndarray): An integer array of shape
            :math:`(1, height, width)`.
            The values correspond to id for label names stored in
            :obj:`label_names`.
        alpha (float): The value which determines transparency of the figure.
            The range of this value is :math:`[0, 1]`. If this
            value is :obj:`0`, the figure will be completely transparent.
            The default value is :obj:`1`. This option is useful for
            overlaying labels on the source images.
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

    ax.imshow(label[0], vmin=-1, vmax=label.max(), alpha=alpha)

    return ax
