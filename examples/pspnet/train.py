#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import sys
import time
from functools import partial
from importlib import import_module

import numpy as np
import yaml

import chainer
import chainermn
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extension
from chainer.training import extensions
from chainer.training import triggers
from mpi4py import MPI


class ConfigBase(object):

    def __init__(self, required_keys, optional_keys, kwargs, name):
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(
                    '{} config should have the key {}'.format(name, key))
            setattr(self, key, kwargs[key])
        for key in optional_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])
            elif key == 'args':
                setattr(self, key, {})
            else:
                setattr(self, key, None)


class Dataset(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'module',
            'name',
            'batchsize',
        ]
        optional_keys = [
            'args',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


class Extension(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = []
        optional_keys = [
            'dump_graph',
            'Evaluator',
            'ExponentialShift',
            'LinearShift',
            'LogReport',
            'observe_lr',
            'observe_value',
            'snapshot',
            'PlotReport',
            'PrintReport',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


class Model(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'module',
            'name',
        ]
        optional_keys = [
            'args'
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


class Loss(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'module',
            'name',
        ]
        optional_keys = [
            'args',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


class Optimizer(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'method'
        ]
        optional_keys = [
            'args',
            'weight_decay',
            'lr_drop_ratio',
            'lr_drop_trigger',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


class UpdaterCreator(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'module',
            'name',
        ]
        optional_keys = [
            'args',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


class PolynomialShift(extension.Extension):

    def __init__(self, attr, power, stop_trigger, batchsize, len_dataset):
        self._attr = attr
        self._power = power
        self._init = None
        self._t = 0
        self._last_value = 0

        if stop_trigger[1] == 'iteration':
            self._maxiter = stop_trigger[0]
        elif stop_trigger[1] == 'epoch':
            n_iter_per_epoch = len_dataset / float(batchsize)
            self._maxiter = float(stop_trigger[0] * n_iter_per_epoch)

    def initialize(self, trainer):
        optimizer = trainer.updater.get_optimizer('main')
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

    def __call__(self, trainer):
        self._t += 1

        optimizer = trainer.updater.get_optimizer('main')
        value = self._init * ((1 - (self._t / self._maxiter)) ** self._power)
        setattr(optimizer, self._attr, value)
        self._last_value = value

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)


def get_dataset(module_name, class_name, args):
    print('get_dataset:', module_name, class_name)
    mod = import_module(module_name)
    return getattr(mod, class_name)(**args), mod.__file__


def get_dataset_from_config(config):
    def get_dataset_object(key):
        d = Dataset(**config['dataset'][key])
        dataset, fn = get_dataset(d.module, d.name, d.args)
        bname = os.path.basename(fn)
        shutil.copy(
            fn, '{}/{}_{}'.format(config['result_dir'], key, bname))
        return dataset
    datasets = dict(
        [(key, get_dataset_object(key)) for key in config['dataset']])
    return datasets['train'], datasets['valid']


def get_model(
        result_dir, model_module, model_name, model_args, loss_module,
        loss_name, loss_args, comm):
    mod = import_module(model_module)
    model_file = mod.__file__
    model = getattr(mod, model_name)

    # Copy model file
    if chainer.config.train:
        dst = '{}/{}'.format(result_dir, os.path.basename(model_file))
        if not os.path.exists(dst):
            shutil.copy(model_file, dst)

    # Initialize
    if model_args is not None:
        if 'comm' in model_args:
            model_args['comm'] = comm
        model = model(**model_args)
    else:
        model = model()

    # Wrap with a loss class
    if chainer.config.train and loss_name is not None:
        mod = import_module(loss_module)
        loss_file = mod.__file__
        loss = getattr(mod, loss_name)
        if loss_args is not None:
            model = loss(model, **loss_args)
        else:
            model = loss(model)

        # Copy loss file
        dst = '{}/{}'.format(result_dir, os.path.basename(loss_file))
        if not os.path.exists(dst):
            shutil.copy(loss_file, dst)
    return model


def get_model_from_config(config, comm=None):
    model = Model(**config['model'])
    loss = Loss(**config['loss'])
    return get_model(
        config['result_dir'], model.module, model.name, model.args,
        loss.module, loss.name, loss.args, comm)


def get_optimizer(model, method, optimizer_args, weight_decay=None):
    optimizer = getattr(optimizers, method)(**optimizer_args)
    optimizer.setup(model)
    if weight_decay is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer


def get_optimizer_from_config(model, config):
    opt_config = Optimizer(**config['optimizer'])
    optimizer = get_optimizer(
        model, opt_config.method, opt_config.args, opt_config.weight_decay)
    return optimizer


def get_updater_creator(module, name, args):
    mod = import_module(module)
    updater_creator = getattr(mod, name)
    if args is not None:
        return partial(updater_creator, **args)
    else:
        return updater_creator


def get_updater_creator_from_config(config):
    updater_creator_config = UpdaterCreator(**config['updater_creator'])
    updater_creator = get_updater_creator(
        updater_creator_config.module, updater_creator_config.name,
        updater_creator_config.args)
    return updater_creator


def create_result_dir(prefix='result'):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        result_dir = 'results/{}_{}_0'.format(
            prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
        while os.path.exists(result_dir):
            i = result_dir.split('_')[-1]
            result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        result_dir = None
    result_dir = comm.bcast(result_dir, root=0)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def create_result_dir_from_config_path(config_path):
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return create_result_dir(config_name)


def save_config_get_log_fn(result_dir, config_path):
    save_name = os.path.basename(config_path)
    a, b = os.path.splitext(save_name)
    save_name = '{}_0{}'.format(a, b)
    i = 0
    while os.path.exists('{}/{}'.format(result_dir, save_name)):
        i += 1
        save_name = '{}_{}{}'.format(a, i, b)
    shutil.copy(config_path, '{}/{}'.format(result_dir, save_name))
    return 'log_{}'.format(i)


def create_iterators(train_dataset, valid_dataset, config):
    train = Dataset(**config['dataset']['train'])
    valid = Dataset(**config['dataset']['valid'])
    train_iter = iterators.MultiprocessIterator(
        train_dataset, train.batchsize)
    valid_iter = iterators.MultiprocessIterator(
        valid_dataset, valid.batchsize, repeat=False, shuffle=False)
    return train_iter, valid_iter


def create_updater(train_iter, optimizer, device):
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    return updater


def get_trainer(args):
    config = yaml.load(open(args.config))

    # Set workspace size
    if 'max_workspace_size' in config:
        chainer.cuda.set_max_workspace_size(config['max_workspace_size'])

    # Prepare ChainerMN communicator
    if args.gpu:
        if args.communicator == 'naive':
            print("Error: 'naive' communicator does not support GPU.\n")
            exit(-1)
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        if args.communicator != 'naive':
            print('Warning: using naive communicator '
                  'because only naive supports CPU-only execution')
        comm = chainermn.create_communicator('naive')
        device = -1

    # Show the setup information
    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
        if args.gpu:
            print('Using GPUs - max workspace size:',
                  chainer.cuda.get_max_workspace_size())
        print('Using {} communicator'.format(args.communicator))

    # Output version info
    if comm.rank == 0:
        print('Chainer version: {}'.format(chainer.__version__))
        print('ChainerMN version: {}'.format(chainermn.__version__))
        print('cuda: {}, cudnn: {}'.format(
            chainer.cuda.available, chainer.cuda.cudnn_enabled))

    # Create result_dir
    if args.result_dir is not None:
        config['result_dir'] = args.result_dir
        model_fn = config['model']['module'].split('.')[-1]
        sys.path.insert(0, args.result_dir)
        config['model']['module'] = model_fn
    else:
        config['result_dir'] = create_result_dir_from_config_path(args.config)
    log_fn = save_config_get_log_fn(config['result_dir'], args.config)
    if comm.rank == 0:
        print('result_dir:', config['result_dir'])

    # Instantiate model
    model = get_model_from_config(config, comm)
    if args.gpu:
        chainer.cuda.get_device(device).use()
        model.to_gpu()
    if comm.rank == 0:
        print('model:', model.__class__.__name__)

    # Initialize optimizer
    optimizer = get_optimizer_from_config(model, config)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    if comm.rank == 0:
        print('optimizer:', optimizer.__class__.__name__)

    # Setting up datasets
    if comm.rank == 0:
        train_dataset, valid_dataset = get_dataset_from_config(config)
        print('train_dataset: {}'.format(len(train_dataset)),
              train_dataset.__class__.__name__)
        print('valid_dataset: {}'.format(len(valid_dataset)),
              valid_dataset.__class__.__name__)
    else:
        train_dataset, valid_dataset = [], []
    train_dataset = chainermn.scatter_dataset(train_dataset, comm)
    valid_dataset = chainermn.scatter_dataset(valid_dataset, comm)

    # Create iterators
    # multiprocessing.set_start_method('forkserver')
    train_iter, valid_iter = create_iterators(
        train_dataset, valid_dataset, config)
    if comm.rank == 0:
        print('train_iter:', train_iter.__class__.__name__)
        print('valid_iter:', valid_iter.__class__.__name__)

    # Create updater and trainer
    if 'updater_creator' in config:
        updater_creator = get_updater_creator_from_config(config)
        updater = updater_creator(train_iter, optimizer, device=device)
    else:
        updater = create_updater(train_iter, optimizer, device=device)
    if comm.rank == 0:
        print('updater:', updater.__class__.__name__)

    # Create Trainer
    trainer = training.Trainer(
        updater, config['stop_trigger'], out=config['result_dir'])
    if comm.rank == 0:
        print('Trainer stops:', config['stop_trigger'])

    # Trainer extensions
    for ext in config['trainer_extension']:
        ext, values = ext.popitem()
        if ext == 'LogReport' and comm.rank == 0:
            trigger = values['trigger']
            trainer.extend(extensions.LogReport(
                trigger=trigger, log_name=log_fn))
        elif ext == 'observe_lr' and comm.rank == 0:
            trainer.extend(extensions.observe_lr(), trigger=values['trigger'])
        elif ext == 'dump_graph' and comm.rank == 0:
            trainer.extend(extensions.dump_graph(**values))
        elif ext == 'Evaluator':
            assert 'module' in values
            mod = import_module(values['module'])
            evaluator = getattr(mod, values['name'])
            if evaluator is extensions.Evaluator:
                evaluator = evaluator(valid_iter, model, device=device)
            else:
                evaluator = evaluator(valid_iter, model.predictor)
            evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
            trainer.extend(
                evaluator, trigger=values['trigger'], name=values['prefix'])
        elif ext == 'PlotReport' and comm.rank == 0:
            trainer.extend(extensions.PlotReport(**values))
        elif ext == 'PrintReport' and comm.rank == 0:
            trigger = values.pop('trigger')
            trainer.extend(extensions.PrintReport(**values),
                           trigger=trigger)
        elif ext == 'ProgressBar' and comm.rank == 0:
            upd_int = values['update_interval']
            trigger = values['trigger']
            trainer.extend(extensions.ProgressBar(
                update_interval=upd_int), trigger=trigger)
        elif ext == 'snapshot' and comm.rank == 0:
            filename = values['filename']
            trigger = values['trigger']
            trainer.extend(extensions.snapshot(
                filename=filename), trigger=trigger)

    # LR decay
    if 'lr_drop_ratio' in config['optimizer'] \
            and 'lr_drop_triggers' in config['optimizer']:
        ratio = config['optimizer']['lr_drop_ratio']
        points = config['optimizer']['lr_drop_triggers']['points']
        unit = config['optimizer']['lr_drop_triggers']['unit']
        drop_trigger = triggers.ManualScheduleTrigger(points, unit)

        def lr_drop(trainer):
            trainer.updater.get_optimizer('main').lr *= ratio
        trainer.extend(lr_drop, trigger=drop_trigger)

    if 'lr_drop_poly_power' in config['optimizer']:
        power = config['optimizer']['lr_drop_poly_power']
        stop_trigger = config['stop_trigger']
        batchsize = train_iter.batch_size
        len_dataset = len(train_dataset)
        trainer.extend(
            PolynomialShift('lr', power, stop_trigger, batchsize, len_dataset),
            trigger=(1, 'iteration'))

    # Resume
    if args.resume is not None:
        # fn = '{}.bak'.format(args.resume)
        # shutil.copy(args.resume, fn)
        serializers.load_npz(args.resume, trainer)
        if comm.rank == 0:
            print('Resumed from:', args.resume)

    if comm.rank == 0:
        print('==========================================')

    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--communicator', type=str, default='single_node')
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    trainer = get_trainer(args)
    trainer.run()
