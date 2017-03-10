import numpy as np


def vis_img_bbox(img, bboxes, label_names=None, ax=None):
    """Visualize bounding boxes inside image

    For notation, `H` and `W` are height and width of a image respectively.
    `R` is the number of bounding boxes in a image.

    Example:

        >>> import chainercv
        >>> import matplotlib.pyplot as plot
        >>> dataset = chainercv.datasets.VOCDetectionDataset()
        >>> img, bboxes = dataset[10]
        >>> img = chainercv.transforms.chw_to_pil_image(img)
        >>> chainercv.tasks.vis_img_bbox(img, bboxes, dataset.labels)
        >>> plot.show()


    Args:
        img (numpy.ndarray): array of shape (H, W, 3). This is in RGB format.
        bboxes (numpy.ndarray): array of shape (R, 5). Elements are organized
            by (x_min, y_min, x_max, y_max, label_id).
        label_names (iterable of strings): name of labels ordered according
            to label_ids. If this is `None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is None, new axis is created.

    """
    from matplotlib import pyplot as plot
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

    for bbox in bboxes:
        xy = (bbox[0], bbox[1])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=3))
        if label_names is not None:
            ax.text(bbox[0], bbox[1], label_names[bbox[4].astype(np.int)],
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
