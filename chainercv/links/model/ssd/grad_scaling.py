from chainer import cuda


class GradScaling(object):

    """Optimizer/UpdateRule hook function for scaling gradient.

    This hook function scales gradient by a constant value.
    Args:
        rate (float): Coefficient for scaling.
    Attributes:
        rate (float): Coefficient for scaling.
    """
    name = 'LrMulti'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        g = param.grad
        with cuda.get_device(g):
            g *= self.rate
