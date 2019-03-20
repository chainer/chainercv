# utility functions taken straight-up from the source project.
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def expand_input_by_factor(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)
