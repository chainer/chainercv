from chainer import variable
from chainer.dataset import convert
from chainer.training import StandardUpdater

import copy
import six


class ParallelUpdater(StandardUpdater):

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 models=None, devices=None, loss_func=None):
        super(ParallelUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            loss_func=loss_func,
        )

        if models is None:
            if devices is None:
                raise ValueError('either models or devices must be specified')
            names = list(six.iterkeys(devices))

            try:
                names.remove('main')
            except ValueError:
                raise KeyError("'devices' must contain a 'main' key.")

            models = {'main': optimizer.target}
            for name in names:
                model = copy.deepcopy(optimizer.target)
                if devices[name] >= 0:
                    model.to_gpu(devices[name])
                models[name] = model
            if devices['main'] >= 0:
                optimizer.target.to_gpu(devices['main'])

        self._devices = devices
        self._models = models

    def connect_trainer(self, trainer):
        # Add observers for all (other) models.
        model_main = self.get_optimizer('main').target
        models_others = {
            k: v for k, v in self._models.items() if v != model_main
        }
        for name, model in models_others.items():
            trainer.reporter.add_observer(name, model)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}

        batch = self.get_iterator('main').next()

        #
        # Split the batch to sub-batches.
        #
        n = len(self._models)
        in_arrays_list = {}
        for i, key in enumerate(six.iterkeys(self._models)):
            in_arrays_list[key] = self.converter(
                batch[i::n], self._devices[key])

        # For reducing memory
        for model in six.itervalues(self._models):
            model.cleargrads()

        losses = []
        for model_key, model in six.iteritems(self._models):
            in_arrays = in_arrays_list[model_key]
            loss_func = self.loss_func or model

            if isinstance(in_arrays, tuple):
                in_vars = tuple(variable.Variable(x) for x in in_arrays)
                losses.append(loss_func(*in_vars))
            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x)
                           for key, x in six.iteritems(in_arrays)}
                losses.append(loss_func(**in_vars))
            else:
                in_vars = variable.Variable(in_arrays)
                losses.append(loss_func(in_vars))

        for i in range(4):
            # For _uninitialized_params
            for model in six.itervalues(self._models):
                model.zerograds()

            for loss in losses:
                loss[i].backward()

            for model in six.itervalues(models_others):
                model_main.addgrads(model)

            optimizer.update()

            for model in six.itervalues(models_others):
                model.copyparams(model_main)
