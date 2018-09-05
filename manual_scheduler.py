from chainer.training import extension


class ManualScheduler(extension.Extension):

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
