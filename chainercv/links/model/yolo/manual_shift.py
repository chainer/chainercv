import chainer


class ManualShift(chainer.training.extension.Extension):

    def __init__(self, attr, values, init=None, optimizer=None):
        self._attr = attr
        self._values = [init] + list(values)
        self._optimizer = optimizer
        self._t = 0

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        if self._values[0] is None:
            self._values[0] = getattr(optimizer, self._attr)
        setattr(optimizer, self._attr, self._values[self._t])

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        setattr(optimizer, self._attr, self._values[self._t])

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')
