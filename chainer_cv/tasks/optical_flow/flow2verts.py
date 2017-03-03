import numpy as np

BIG_NUMBER = 100000


def flow2verts(flow):
    """Convert Flow representation to vertices representation.

    Args:
        flow array (H, W, 3)

    Returns:
        vert_pairs: array of shape (2, n_verts, 2).
            Left image corresponds to the first index of the first axis.
            Right image corresponds to the second index.

            Coordinates are represented as (v, u) where u goes horizontally
            and v goes vertically. The top-left is the zero.
    """
    H, W, _ = flow.shape
    src_verts = np.stack(np.where(flow > 0)[:2], axis=1)  # (n_verts, 2)
    dest_verts = []
    for vert in src_verts:
        dest_verts.append([vert[0] + flow[vert[0], vert[1], 0],
                           vert[1] + flow[vert[0], vert[1], 1]])
    dest_verts = np.array(dest_verts)
    vert_pairs = np.stack([src_verts, dest_verts])
    h_verts = np.clip(np.round(vert_pairs[:, :, 0]), 0, H - 1)
    w_verts = np.clip(np.round(vert_pairs[:, :, 1]), 0, W - 1)
    return np.stack([h_verts, w_verts], axis=2).astype(np.int)
