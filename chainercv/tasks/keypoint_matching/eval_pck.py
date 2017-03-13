import numpy as np


def eval_pck(pred, expected, alpha, L):
    """Calculate PCK (Percentage of Correct Keypoints).

    This function calculates number of vertices whose positions are correctly
    pred.
    A pred keypoint is correctly matched to the ground truth if it lies
    within Euclidean distance :math:`\\alpha \\cdot L` of the ground truth
    keypoint, where :math:`L` is the size of the image and
    :math:`0 < \\alpha < 1` is a variable we control.
    :math:`L` is determined differently depending on the context. For example,
    in evaluation of keypoint matching for CUB dataset,
    :math:`L=\\sqrt{w^2 + h^2}` is used.

    Args:
        pred (~numpy.ndarray): An array of shape :math:`(N, 2)` or
            :math:`(N, 3)`, where
            :math:`N` is the number of vertices to be evaluated. The
            first two elements of the second axis corresponds to x and y
            coordinate of the keypoint. If the shape is :math:`(N, 3)`,
            the last element corresponds to whether the point should be
            included in calculation or not. If either of corresponding
            vertices in :obj:`pred` and :obj:`expected` is specified to
            be excluded from the calculation, that pair will not be used
            for evaluation.
        expected (~numpy.ndarray): Same kind of array as :obj:`pred`.
            This contains ground truth location of the keypoints that
            the user tries to predict.
        alpha (float): A control variable :math:`\\alpha`.
        L (float): A size of an image. The definition changes from the tasks to
            task.

    Returns:
        float
    """
    m = pred.shape[0]
    n = expected.shape[0]
    assert m == n

    if pred.shape[1] == 3 and expected.shape[1] == 3:
        verts = np.stack([pred, expected], axis=1)  # (N, 2, 3)
        valid_mask = np.all(verts[:, :, 2] == 1, axis=1)
        verts_idx, = np.where(valid_mask)
        pred = pred[verts_idx]
        expected = expected[verts_idx]

    difference = np.linalg.norm(pred[:, :2] - expected[:, :2], axis=1)

    pck_accuracy = np.mean(difference < alpha * L)
    return pck_accuracy
