from __future__ import division

import sys
import time


class ProgressHook(object):
    """A hook class reporting the progress of iteration.

    This is a hook class designed for
    :func:`~chainercv.utils.apply_prediction_to_iterator`.

    Args:
        n_total (int): The number of images. This argument is optional.
    """

    def __init__(self, n_total=None):
        self.n_total = n_total
        self.start = time.time()
        self.n_processed = 0

    def __call__(self, in_values, out_values, rest_values):
        self.n_processed += len(in_values[0])
        fps = self.n_processed / (time.time() - self.start)
        if self.n_total is not None and fps > 0:
            eta = int((self.n_total - self.n_processed) / fps)
            sys.stdout.write(
                '\r{:d} of {:d} samples, {:.2f} samples/sec,'
                ' ETA {:4d}:{:02d}:{:02d}'.format(
                    self.n_processed, self.n_total, fps,
                    eta // 60 // 60, (eta // 60) % 60, eta % 60))
        else:
            sys.stdout.write(
                '\r{:d} samples, {:.2f} samples/sec'.format(
                    self.n_processed, fps))

        sys.stdout.flush()
