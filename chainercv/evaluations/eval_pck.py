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
    :math:`L=\\sqrt{h^2 + w^2}` is used.

    Args:
        pred (~numpy.ndarray): An array of shape :math:`(K, 2)`
            :math:`N` is the number of keypoints to be evaluated. The
            two elements of the second axis corresponds to :math:`y`
            and :math:`x` coordinate of the keypoint.
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

    difference = np.linalg.norm(pred[:, :2] - expected[:, :2], axis=1)

    pck_accuracy = np.mean(difference <= alpha * L)
    return pck_accuracy
