import numpy as np
import random

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from chainer_cv.models import GoogLeNet


class TripletLossEmbedding(chainer.Chain):

    def __init__(self, embed_size=128):
        super(TripletLossEmbedding, self).__init__(
            googlenet=GoogLeNet(),
            fc=L.Linear(1024, embed_size)
        )

    def __call__(self, x):
        h = self.googlenet(x, layers=['pool6'])['pool6']
        return self.fc(h)


class TripletLossIterator(chainer.iterators.SerialIterator):

    def __init__(self, dataset, batch_size, repeat=True):
        assert batch_size % 3 == 0
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        indices = random.sample(range(len(self.dataset)), self.batch_size / 3)
        anchor_batch = [self.dataset[index] for index in indices]

        pos_batch = []
        neg_batch = []
        for i, v in zip(indices, anchor_batch):
            class_id = v[1]
            ids = self.dataset.get_ids(np.asscalar(class_id))
            ids.remove(i)
            pos_batch.append(self.dataset[random.choice(ids)])

            for k in range(100):
                if k == 99:
                    raise ValueError('failed to find a neg pair')
                neg_id = random.choice(range(len(self.dataset)))
                neg = self.dataset[neg_id]
                if class_id != neg[1]:
                    neg_batch.append(neg)
                    break

        batch = anchor_batch + pos_batch + neg_batch

        self.current_position += self.batch_size
        if self.current_position > len(self.dataset):
            self.is_new_epoch = True
            self.epoch += 1
            self.current_position = 0
        return batch

    next = __next__


class TripletLossUpdater(chainer.training.StandardUpdater):

    def __init__(self, iterator, optimizer, device):
        super(TripletLossUpdater, self).__init__(
            iterator, optimizer, device=device)

    def update_core(self):
        batch = self._iterators['main'].next()
        batch_size = len(batch)
        optimizer = self._optimizers['main']
        model = optimizer.target

        in_arrays = self.converter(batch, self.device)
        in_vars = tuple(chainer.Variable(x) for x in in_arrays)

        feats = model(in_vars[0])
        anchor_feats = feats[:batch_size / 3]
        pos_feats = feats[batch_size / 3: batch_size / 3 * 2]
        neg_feats = feats[batch_size / 3 * 2:]

        loss = F.triplet(anchor_feats, pos_feats, neg_feats)
        reporter.report({'loss': loss}, model)

        loss.backward()
        optimizer.update()
        model.cleargrads()
