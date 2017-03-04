import random


def random_flip(xs, orientation='h', return_flip=False):
    force_array = False
    if not isinstance(xs, tuple):
        xs = (xs,)
        force_array = True
    if not isinstance(orientation, list):
        orientation = [orientation]

    h_flip, v_flip = False, False
    if 'h' in orientation:
        h_flip = random.choice([True, False])
    if 'v' in orientation:
        v_flip = random.choice([True, False])

    outs = []
    for x in xs:
        if h_flip:
            x = x[:, :, ::-1]
        if v_flip:
            x = x[:, ::-1, :]
        outs.append(x)

    if force_array:
        outs = outs[0]
    else:
        outs = tuple(outs)

    if return_flip:
        return outs, {'h': h_flip, 'v': v_flip}
    else:
        return outs
