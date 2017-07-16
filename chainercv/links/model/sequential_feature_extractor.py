import chainer


class SequentialFeatureExtractor(chainer.Chain):

    """A feature extractor model with a single-stream forward pass.

    This class is a base class that can be used for an implementation of
    a feature extractor model.
    Callable objects, such as :class:`chainer.Link` and
    :class:`chainer.Function`, can be registered to this chain with
    :meth:`init_scope`.
    This chain keeps the order of registerations and :meth:`__call__`
    executes callables in that order.
    A :class:`chainer.Link` object in the sequence will be added as
    a child link of this link.

    :meth:`__call__` returns single or multiple features that are picked up
    through a stream of computation.
    These features can be specified by :obj:`feature_names`, which contains
    the names of the features that are collected.
    When :obj:`feature_names` is a string, single feature is returned.
    When :obj:`feature_names` is an iterable of strings, a tuple of features
    is returned. The order of the features is the same as the order of
    the strings in :obj:`feature_names`.
    When :obj:`feature_names` is :obj:`None`, the last feature is returned.

    Examples:

        >>> import chainer.functions as F
        >>> import chainer.links as L
        >>> model = SequentialFeatureExtractor()
        >>> with model.init_scope():
        >>>     model.l1 = L.Linear(None, 1000)
        >>>     model.l1_relu = F.relu
        >>>     model.l2 = L.Linear(None, 1000)
        >>>     model.l2_relu = F.relu
        >>>     model.l3 = L.Linear(None, 10)
        >>> # This is feature l3
        >>> feat3 = model(x)
        >>> # The features to be collected can be changed.
        >>> model.feature_names = ('l2_relu', 'l1_relu')
        >>> # These are features l2_relu and l1_relu.
        >>> feat2, feat1 = model(x)

    Params:
        feature_names (string or iterable of strings):
            Names of features that are collected during
            the forward pass.

    """

    def __init__(self):
        super(SequentialFeatureExtractor, self).__init__()
        self._order = list()
        self.feature_names = None
        # Two attributes are initialized by the setter of feature_names.
        # self._feature_names -> None
        # self._return_tuple -> False

    def __setattr__(self, name, value):
        super(SequentialFeatureExtractor, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self._order.append(name)

    def __delattr__(self, name):
        if self._feature_names and name in self._feature_names:
            raise AttributeError(
                'Feature {:s} is registered to feature_names.'.format(name))
        super(SequentialFeatureExtractor, self).__delattr__(name)
        try:
            self._order.remove(name)
        except ValueError:
            pass

    @property
    def feature_names(self):
        if self._feature_names is None:
            return None

        if self._return_tuple:
            return self._feature_names
        else:
            return self._feature_names[0]

    @feature_names.setter
    def feature_names(self, feature_names):
        if feature_names is None:
            self._return_tuple = False
            self._feature_names = None
            return

        if (not isinstance(feature_names, str) and
                all(isinstance(name, str) for name in feature_names)):
            return_tuple = True
        else:
            return_tuple = False
            feature_names = (feature_names,)
        if any(name not in self._order for name in feature_names):
            raise ValueError('Invalid feature name')

        self._return_tuple = return_tuple
        self._feature_names = tuple(feature_names)

    def __call__(self, x):
        """Forward this model.

        Args:
            x (chainer.Variable or array): Input to the model.

        Returns:
            chainer.Variable or tuple of chainer.Variable:
            The returned features are determined by :obj:`feature_names`.

        """
        if self._feature_names is None:
            feature_names = (self._order[-1],)
        else:
            feature_names = self._feature_names

        # The biggest index among indices of the features that are included
        # in feature_names.
        last_index = max(self._order.index(name) for name in feature_names)

        features = {}
        h = x
        for name in self._order[:last_index + 1]:
            h = self[name](h)
            if name in feature_names:
                features[name] = h

        if self._return_tuple:
            features = tuple(features[name] for name in feature_names)
        else:
            features = list(features.values())[0]
        return features
