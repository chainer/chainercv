import numpy as np

import chainer


class ManualShift(chainer.training.extension.Extension):

    def __init__(self, attr, values, init=None, target=None, optimizer=None):
        self._attr = attr
        self._values = values
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        self._update_value(optimizer, self._values[self._t])

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
