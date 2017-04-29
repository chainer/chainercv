import chainer


class ConstantReturnModel(chainer.Chain):
    """A chainer.Chain that returns constant values

    This is a ``chainer.Chain`` instance that returns `return_value` when
    a method `__call__` is called.

    """

    def __init__(self, return_value):
        super(ConstantReturnModel, self).__init__()
        self.return_value = return_value

    def __call__(self, *args):
        return self.return_value
