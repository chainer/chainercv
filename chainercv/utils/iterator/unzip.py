import collections


class BufferedIterator(object):
    """Buffered iterator for :func:`unzip`.

    This iterator contains :obj:`buffers` and :obj:`index`.
    The buffers are shared with other :class:`BufferedIterator`s.
    When :method:`__next__` or :method:`next` is called,
    this iterator checks :obj:`buffers[index]` fisrt.
    If :obj:`buffers[index]` has some values, it pops
    the first value and returns it. Otherwise, it gets
    a new tuple from the base iterator and pushes the values
    into :obj:`buffers`.

    When this iterator is deleted, it disables the corresponding buffer
    by setting :obj:`buffers[index]` to :obj:`None`.
    With this mark, other iterators can skip values for this deleted
    iterator and memory usage can be reduced.

    Args:
        iterator (iterator): A base iterator of tuples. All tuples should have
            the same length.
        buffers (list of collections.deque): A list of
            :class:`collections.deque` to buffer values.
            The size of this list should be same as those of tuples
            from :obj:`iterator`.
        index (int): The index of this :class:`BufferedIterator`.

    """

    def __init__(self, iterator, buffers, index):
        self.iterator = iterator
        self.buffers = buffers
        self.index = index

    def __del__(self):
        self.buffers[self.index] = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.buffers[self.index].popleft()
        except IndexError:
            values = next(self.iterator)
            for buf, val in zip(self.buffers, values):
                # skip a value if the correponding iterator is deleted.
                if buf is not None:
                    buf.append(val)
            return self.buffers[self.index].popleft()

    next = __next__


def unzip(iterable):
    """Converts an iterable of tuples into a tuple of iterators.

    This function converts an iterable of tuples into a tuple of iterators.
    This is an inverse function of :func:`six.moves.zip`.

    >>> from chainercv.utils import unzip
    >>> data = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]
    >>> int_iter, str_iter = unzip(data)
    >>>
    >>> next(int_iter)  # 0
    >>> next(int_iter)  # 1
    >>> next(int_iter)  # 2
    >>>
    >>> next(str_iter)  # 'a'
    >>> next(str_iter)  # 'b'
    >>> next(str_iter)  # 'c'

    Args:
        iterable (iterable): An iterable of tuples. All tuples should have
            the same length.

    Returns:
        tuple of iterators:
        Each iterator corresponds to each element of input tuple.
        Note that each iterator stores values until they are popped.
        To reduce memory usage, it is recommended to delete unused iterators.
    """

    iterator = iter(iterable)
    values = next(iterator)
    buffers = [collections.deque((val,)) for val in values]
    return tuple(
        BufferedIterator(iterator, buffers, index)
        for index in range(len(buffers)))
