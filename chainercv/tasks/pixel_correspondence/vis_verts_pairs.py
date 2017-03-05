import numpy as np

import matplotlib.pyplot as plt


def vis_verts_pairs(src_img, dst_img, verts, axes=None):
    """Visualize vertex pairs

    Args:
        src_img (H, W, 3)
        dst_img (H, W, 3)
        verts (n_verts, 2, 3)  (x, y, valid) or (n_verts, 2, 2)

        axes (length two list of matplotlib.axes.Axes)

    """

    if axes is None:
        fig = plt.figure()
        axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    assert len(axes) == 2

    n_colors = 15
    cm = plt.get_cmap('gist_rainbow')

    colors = [cm(1. * i / n_colors) for i in range(n_colors)]
    if verts.shape[2] == 3:
        valid_indices = np.where(np.all(verts[:, :, 2] > 0, axis=1))[0]
    elif verts.shape[2] == 2:
        valid_indices = range(verts.shape[1])
    else:
        raise ValueError('invalid vertex shape')
    select_idxs = np.random.choice(
        valid_indices,
        size=(min(len(colors), len(valid_indices)),), replace=False)
    select_idxs = np.sort(select_idxs)

    src_verts = verts[:, 0]
    dst_verts = verts[:, 1]
    axes[0].imshow(src_img)
    for i, idx in enumerate(select_idxs):
        axes[0].scatter(src_verts[idx, 0], src_verts[idx, 1],
                        c=colors[i], s=100)

    axes[1].imshow(dst_img)
    for i, idx in enumerate(select_idxs):
        axes[1].scatter(dst_verts[idx, 0], dst_verts[idx, 1],
                        c=colors[i], s=100)
