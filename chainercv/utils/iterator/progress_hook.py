from __future__ import division

import sys
import time


class ProgressHook(object):
    """A progress hook for chainercv.utils.apply_prediction_to_iterator

    This is a hook class designed for
    :func:`~chainercv.utils.apply_prediction_to_iterator`.

    Args:
        n_total (int): The number of images. This argument is optional.
    """

    def __init__(self, n_total=None):
        self.n_total = n_total
        self.start = time.time()
        self.n_processed = 0

    def __call__(self, imgs, pred_values, gt_values):
        self.n_processed += len(imgs)
        fps = self.n_processed / (time.time() - self.start)
        if self.n_total is not None:
            sys.stdout.write(
                '\r{:d} of {:d} images, {:.2f} FPS'.format(
                    self.n_processed, self.n_total, fps))
        else:
            sys.stdout.write(
                '\r{:d} images, {:.2f} FPS'.format(
                    self.n_processed, fps))

        sys.stdout.flush()
