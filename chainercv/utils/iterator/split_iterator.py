class IteratorHub(object):

    def __init__(self, iterator):
        self.iterator = iterator

        sample = next(self.iterator)
        self.children = tuple(BufferedIterator(self) for _ in sample)
        for child, value in zip(self.children, sample):
            child.append(value)

    def feed(self):
        sample = next(self.iterator)
        for child, value in zip(self.children, sample):
            child.append(value)


class BufferedIterator(object):

    def __init__(self, hub):
        self.hub = hub
        self.buffer = list()

    def append(self, value):
        self.buffer.append(value)

    def __next__(self):
        try:
            return self.buffer.pop(0)
        except:
            self.hub.feed()
            return self.buffer.pop(0)


def split_iterator(iterator):
    hub = IteratorHub(iterator)
    return hub.children
