import numpy as np

from matplotlib import pyplot as plt


def vis_img_bbox(img, bboxes, label_names=None, ax=None):
    """Visualize bounding boxes inside image

    For notation, `H` and `W` are height and width of a image respectively.
    `R` is the number of bounding boxes in a image.

    Args:
        img (numpy.ndarray): array of shape (H, W, 3). This is in RGB format.
        bboxes (numpy.ndarray): array of shape (R, 5). Elements are organized
            by (x_min, y_min, x_max, y_max, label_id).
        label_names (iterable of strings): name of labels ordered according
            to label_ids. If this is `None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is None, new axis is created.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

    for bbox in bboxes:
        xy = (bbox[0], bbox[1])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=3))
        if label_names is not None:
            ax.text(bbox[0], bbox[1], label_names[bbox[4].astype(np.int)],
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})


if __name__ == '__main__':
    from chainer_cv.datasets import VOCDetectionDataset

    dataset = VOCDetectionDataset()

    for i in range(0, 100, 4):
        fig = plt.figure(figsize=(14, 14))
        for j in range(4):
            ax = fig.add_subplot(2, 2, j + 1)
            img, bboxes = dataset.get_raw_data(i + j)
            vis_img_bbox(img, bboxes, dataset.labels, ax)
        plt.show()
