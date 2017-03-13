import numpy as np
import six


def vis_keypoint_pair(src_img, dst_img, keypoints, axes=None):
    """Visualize keypoint pairs.

    Args:
        src_img (~numpy.ndarray): source mage of shape
            :math:`(height, width, 3)`.
        dst_img (~numpy.ndarray): target image of shape
            :math:`(height, width, 3)`.
        keypoints (~numpy.ndarray): An array with keypoint pairs whose shape is
            either :math:`(K, 2, 3)` or :math:`(K, 2, 2)`, where :math:`K` is
            the number of keypoint pairs contained in the array. The second
            axis corresponds to the source image's keypoints and the
            destination image's keypoints in this order. If :obj:`keypoints`
            has shape :math:`(K, 2, 3)`, then the last axis represents
            :math:`(x, y, valid)`, which are x and y coordinates of the
            keypoint and whether the keypoint is visible or not.
        axes (length two list of matplotlib.axes.Axes or :obj:`None`)

    """
    import matplotlib.pyplot as plot

    if axes is None:
        fig = plot.figure()
        axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    assert len(axes) == 2

    n_colors = 15
    cm = plot.get_cmap('gist_rainbow')

    colors = [cm(1. * i / n_colors) for i in six.moves.range(n_colors)]
    if keypoints.shape[2] == 3:
        valid_indices = np.where(np.all(keypoints[:, :, 2] > 0, axis=1))[0]
    elif keypoints.shape[2] == 2:
        valid_indices = list(six.moves.range(keypoints.shape[1]))
    else:
        raise ValueError('invalid vertex shape')
    select_idxs = np.random.choice(
        valid_indices,
        size=(min(len(colors), len(valid_indices)),), replace=False)
    select_idxs = np.sort(select_idxs)

    src_keypoints = keypoints[:, 0]
    dst_keypoints = keypoints[:, 1]
    axes[0].imshow(src_img)
    for i, idx in enumerate(select_idxs):
        axes[0].scatter(src_keypoints[idx, 0], src_keypoints[idx, 1],
                        c=colors[i], s=100)

    axes[1].imshow(dst_img)
    for i, idx in enumerate(select_idxs):
        axes[1].scatter(dst_keypoints[idx, 0], dst_keypoints[idx, 1],
                        c=colors[i], s=100)
