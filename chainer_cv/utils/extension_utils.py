import numpy as np
import six

import chainer
from chainer.utils import type_check


def forward(model, inputs, forward_func=None, expand_dim=False):
    """Forward model with given inputs

    Args:
        model (chainer.Chain):
            If model stores its paramters in a GPU, the GPU will be used
            for forwarding.
        inputs: tuple of numpy.ndarray to be used as input. If `expand_dim`
            is True, the first axis will be added.
        forward_func (callable): called to forward
        expand_dim (bool)

    Returns:
        tuple of outputs

    """

    if forward_func is None:
        forward_func = model
    input_vars = []
    outputs = []
    xp = model.xp
    for a in inputs:
        if not isinstance(a, np.ndarray):
            raise ValueError('input has to be ndarray')
        if expand_dim:
            a = np.expand_dims(a, axis=0)

        a_var = chainer.Variable(a)
        if xp != np:
            a_var.to_gpu()
        input_vars.append(a_var)

    output_vars = forward_func(*input_vars)
    if not isinstance(output_vars, tuple):
        output_vars = (output_vars,)
    for out in output_vars:
        if isinstance(out, chainer.Variable):
            out = out.data
        out = chainer.cuda.to_cpu(out)
        outputs.append(out)
    return tuple(outputs)


def check_type(check_type_func, name=None):
    """Decorator around a function that checks type of the `in_data`.

    The wrapped function takes two arguments `self` and `in_data`.
    `self` is an Extension class that is being called. `in_data` is a
    tuple or an array-like object whose types are checked.

    """

    def wrapper(self, in_data):
        # force input to be a tuple
        if not isinstance(in_data, tuple):
            _in_data = (in_data,)
        else:
            _in_data = in_data

        # turn input to arrays if they are not
        if any([not isinstance(in_data_i, np.ndarray) and
                not isinstance(in_data_i, chainer.cuda.ndarray) for
                in_data_i in in_data]):
            _in_data = list(_in_data)
            for i, in_data_i in enumerate(_in_data):
                if (not isinstance(in_data_i, np.ndarray) and
                        not isinstance(in_data_i, chainer.cuda.ndarray)):
                    _in_data[i] = np.array(in_data_i)
            _in_data = tuple(_in_data)
        else:
            _in_data = _in_data

        # get types of in_data
        in_types = type_check.get_types(_in_data, 'in_types', False)
        # check types
        try:
            check_type_func(self, in_types)
        except type_check.InvalidType as e:
            msg = """
    Invalid operation is performed in: {0}

    {1}""".format(name, str(e))
            six.raise_from(
                type_check.InvalidType(e.expect, e.actual, msg=msg), None)
    return wrapper
