import numpy as np

import chainer


def forward(model, inputs, expand_dim=True, forward_func=None):
    """Forward model with given inputs

    Args:
        model (chainer.Chain):
            If model stores its paramters in a GPU, the GPU will be used
            for forwarding.
        inputs: tuple of numpy.ndarray to be used as input. If `expand_dim`
            is True, the first axis will be added.
        expand_dim (bool)
        forward_func (callable): called to forward
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
    for a_var in output_vars:
        out = a_var.data
        if xp != np:
            out = chainer.cuda.to_cpu(out)
        outputs.append(out)
    return outputs
