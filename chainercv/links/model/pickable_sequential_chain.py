import copy

import chainer


class PickableSequentialChain(chainer.Chain):

    """A sequential chain that can pick intermediate layers.

    Callable objects, such as :class:`chainer.Link` and
    :class:`chainer.Function`, can be registered to this chain with
    :meth:`init_scope`.
    This chain keeps the order of registrations and :meth:`forward`
    executes callables in that order.
    A :class:`chainer.Link` object in the sequence will be added as
    a child link of this link.

    :meth:`forward` returns single or multiple layers that are picked up
    through a stream of computation.
    These layers can be specified by :obj:`pick`, which contains
    the names of the layers that are collected.
    When :obj:`pick` is a string, single layer is returned.
    When :obj:`pick` is an iterable of strings, a tuple of layers
    is returned. The order of the layers is the same as the order of
    the strings in :obj:`pick`.
    When :obj:`pick` is :obj:`None`, the last layer is returned.

    Examples:

        >>> import chainer.functions as F
        >>> import chainer.links as L
        >>> model = PickableSequentialChain()
        >>> with model.init_scope():
        >>>     model.l1 = L.Linear(None, 1000)
        >>>     model.l1_relu = F.relu
        >>>     model.l2 = L.Linear(None, 1000)
        >>>     model.l2_relu = F.relu
        >>>     model.l3 = L.Linear(None, 10)
        >>> # This is layer l3
        >>> layer3 = model(x)
        >>> # The layers to be collected can be changed.
        >>> model.pick = ('l2_relu', 'l1_relu')
        >>> # These are layers l2_relu and l1_relu.
        >>> layer2, layer1 = model(x)

    Parameters:
        pick (string or iterable of strings):
            Names of layers that are collected during
            the forward pass.
        layer_names (iterable of strings):
            Names of layers that can be collected from
            this chain. The names are ordered in the order
            of computation.

    """

    def __init__(self):
        super(PickableSequentialChain, self).__init__()
        self.layer_names = []
        # Two attributes are initialized by the setter of pick.
        # self._pick -> None
        # self._return_tuple -> False
        self.pick = None

    def __setattr__(self, name, value):
        super(PickableSequentialChain, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self.layer_names.append(name)

    def __delattr__(self, name):
        if self._pick and name in self._pick:
            raise AttributeError(
                'layer {:s} is registered to pick.'.format(name))
        super(PickableSequentialChain, self).__delattr__(name)
        try:
            self.layer_names.remove(name)
        except ValueError:
            pass

    @property
    def pick(self):
        if self._pick is None:
            return None

        if self._return_tuple:
            return self._pick
        else:
            return self._pick[0]

    @pick.setter
    def pick(self, pick):
        if pick is None:
            self._return_tuple = False
            self._pick = None
            return

        if (not isinstance(pick, str) and
                all(isinstance(name, str) for name in pick)):
            return_tuple = True
        else:
            return_tuple = False
            pick = (pick,)
        for name in pick:
            if name not in self.layer_names:
                raise ValueError('Invalid layer name ({:s})'.format(name))

        self._return_tuple = return_tuple
        self._pick = tuple(pick)

    def remove_unused(self):
        """Delete all layers that are not needed for the forward pass.

        """
        if self._pick is None:
            return

        # The biggest index among indices of the layers that are included
        # in pick.
        last_index = max(self.layer_names.index(name) for name in self._pick)
        for name in self.layer_names[last_index + 1:]:
            delattr(self, name)

    def forward(self, x):
        """Forward this model.

        Args:
            x (chainer.Variable or array): Input to the model.

        Returns:
            chainer.Variable or tuple of chainer.Variable:
            The returned layers are determined by :obj:`pick`.

        """
        if self._pick is None:
            pick = (self.layer_names[-1],)
        else:
            pick = self._pick

        # The biggest index among indices of the layers that are included
        # in pick.
        last_index = max(self.layer_names.index(name) for name in pick)

        layers = {}
        h = x
        for name in self.layer_names[:last_index + 1]:
            h = self[name](h)
            if name in pick:
                layers[name] = h

        if self._return_tuple:
            layers = tuple(layers[name] for name in pick)
        else:
            layers = list(layers.values())[0]
        return layers

    def copy(self, *args, **kargs):
        copied = super(PickableSequentialChain, self).copy(*args, **kargs)
        copied.layer_names = copy.copy(self.layer_names)
        copied._pick = copy.copy(self._pick)
        copied._return_tuple = copy.copy(self._return_tuple)

        return copied
