import copy

import chainer


class SequentialFeatureExtractionChain(chainer.Chain):

    def __init__(self, feature_names, link_generators):
        if (not isinstance(feature_names, str) and
                all([isinstance(feature, str) for feature in feature_names])):
            return_tuple = True
        else:
            return_tuple = False
            feature_names = [feature_names]
        self._return_tuple = return_tuple
        self._feature_names = feature_names

        super(SequentialFeatureExtractionChain, self).__init__()

        if any([name not in self.functions for
                name in self._feature_names]):
            raise ValueError('Elements of `feature_names` shuold be one of '
                             '{}.'.format(self.functions.keys()))

        # Remove all functions that are not necessary.
        self._unused_function_names = []
        pop_funcs = False
        features = list(self._feature_names)
        for name in self.functions.keys():
            if pop_funcs:
                self._unused_function_names.append(name)

            if name in features:
                features.remove(name)
            if len(features) == 0:
                pop_funcs = True

        with self.init_scope():
            for name, link_gen in link_generators.items():
                # Ignore layers whose names match functions that are removed.
                if name not in self._unused_function_names:
                    setattr(self, name, link_gen())

    @property
    def functions(self):
        raise NotImplementedError

    def __call__(self, x):
        """Forward the model.

        Args:
            x (~chainer.Variable): Batch of image variables.

        Returns:
            Variable or tuple of Variable:
            A batch of features or tuple of batched features.
            The returned features are selected by :obj:`feature_names` that
            is passed to :meth:`__init__`.

        """
        functions = copy.copy(self.functions)
        for name in self._unused_function_names:
            functions.pop(name)

        features = {}
        h = x
        for name, funcs in functions.items():
            for func in funcs:
                h = func(h)
            if name in self._feature_names:
                features[name] = h

        if self._return_tuple:
            features = tuple(
                [features[name] for name in features.keys()])
        else:
            features = list(features.values())[0]
        return features
