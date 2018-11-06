from chainer.training import extension


class ManualScheduler(extension.Extension):

    """Trainer extension to update an optimizer attribute manually.

    This extension calls :obj:`func` for each invocation and
    set the returned value to the specified attribute of the optimizer.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the attribute to shift.
        func (callable): Callback function that returns the new value of
            the attribute. :obj:`updater` is passed as its argument.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, func, optimizer=None):
        self._attr = attr
        self._func = func
        self._optimizer = optimizer

    def initialize(self, trainer):
        self(trainer)

    def __call__(self, trainer):
        optimizer = self._get_optimizer(trainer)
        setattr(optimizer, self._attr, self._func(trainer.updater))

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')
