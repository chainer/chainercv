import numpy as np
import six

from chainer.utils import type_check
import chainer


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
    """
    This is a decorator for a class method.
    """

    def wrapper(self, in_data):
        if any([not isinstance(in_data_i, np.ndarray) and
                not isinstance(in_data_i, chainer.cuda.ndarray) for
                in_data_i in in_data]):
            in_data_tmp = list(in_data)
            for i, in_data_i in enumerate(in_data):
                if (not isinstance(in_data_i, np.ndarray) and
                        not isinstance(in_data_i, chainer.cuda.ndarray)):
                    in_data_tmp[i] = np.array(in_data_i)
            in_data_tmp = tuple(in_data_tmp)
        else:
            in_data_tmp = in_data

        in_types = type_check.get_types(in_data_tmp, 'in_types', False)
        try:
            check_type_func(self, in_types)
        except type_check.InvalidType as e:
            msg = """
    Invalid operation is performed in: {0}

    {1}""".format(name, str(e))
            six.raise_from(
                type_check.InvalidType(e.expect, e.actual, msg=msg), None)
    return wrapper
