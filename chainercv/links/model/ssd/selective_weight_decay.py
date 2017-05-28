from chainer import cuda


class SelectiveWeightDecay(object):

    name = 'SelectiveWeightDecay'

    def __init__(self, decay, **params):
        self.decay = decay
        self.params = params

    def _kernel(self):
        return cuda.elementwise(
            'T p, T lr, T decay', 'T g',
            'g = lr * g + decay * p',
            'selective_weight_decay')

    def __call__(self, opt):
        for param in opt.target.params():
            lr = 1
            decay = self.decay

            if param.name in self.params:
                if 'lr' in self.params[param.name]:
                    lr = self.params[param.name]['lr']
                if 'decay' in self.params[param.name]:
                    decay *= self.params[param.name]['decay']

            p, g = param.data, param.grad
            with cuda.get_device(p) as dev:
                if int(dev) == -1:
                    g = lr * g + decay * p
                else:
                    self._kernel()(p, lr, decay, g)
