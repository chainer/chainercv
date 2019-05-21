"""Config generator for PFN CI

Usage:
    $ python gen_config.py > config.pbtxt
"""

from collections import OrderedDict
import itertools


def test_config(python, chainer, optional, target):
    key = 'chainercv.py{}.{}'.format(python, chainer)
    if not optional:
        key += '.mini'

    value = OrderedDict((
        ('requirement', OrderedDict((
            ('cpu', 4),
            ('memory', 24),
            ('disk', 10),
        ))),
        ('time_limit', None),
        ('command', None),
        ('environment_variables', [
            ('PYTHON', str(python)),
            ('CHAINER', chainer),
            ('OPTIONAL_MODULES', '1' if optional else '0'),
        ]),
    ))

    if target == 'cpu':
        value['requirement']['cpu'] = 6
        value['requirement']['memory'] = 36
        value['time_limit'] = {'seconds': 3600}
        value['command'] = 'sh .pfnci/tests.sh'
        value['quickfix_checkout_dot_git'] = True
    elif target == 'gpu':
        key += '.gpu'
        value['requirement']['gpu'] = 1
        value['command'] = 'sh .pfnci/tests_gpu.sh'
        value['quickfix_checkout_dot_git'] = True
    elif target == 'examples':
        key += '.examples'
        value['requirement']['gpu'] = 2
        value['time_limit'] = {'seconds': 1800}
        value['command'] = 'sh .pfnci/examples_tests.sh'

    return key, value


def main():
    configs = []

    configs.append((
        'chainercv.cache',
        OrderedDict((
            ('requirement', OrderedDict((
                ('cpu', 8),
                ('memory', 48),
            ))),
            ('time_limit', OrderedDict((
                ('seconds', 1800),
            ))),
            ('command', 'sh .pfnci/cache.sh'),
        ))
    ))

    for python, chainer in itertools.product(
            (2, 3), ('stable', 'latest', 'master')):
        for optional in (True, False):
            configs.append(test_config(python, chainer, optional, 'cpu'))
            configs.append(test_config(python, chainer, optional, 'gpu'))
        configs.append(test_config(python, chainer, True, 'examples'))

    print('# DO NOT MODIFY THIS FILE MANUALLY.')
    print('# USE gen_config.py INSTEAD.')
    print()

    dump_pbtxt('configs', configs)


def dump_pbtxt(key, value, level=0):
    indent = '  ' * level
    if isinstance(value, bool):
        print('{}{}: {}'.format(indent, key, 'true' if value else 'false'))
    elif isinstance(value, int):
        print('{}{}: {}'.format(indent, key, value))
    elif isinstance(value, str):
        print('{}{}: "{}"'.format(indent, key, value))
    elif isinstance(value, list):
        for k, v in value:
            print('{}{} {{'.format(indent, key))
            dump_pbtxt('key', k, level + 1)
            dump_pbtxt('value', v, level + 1)
            print('{}}}'.format(indent))
    elif isinstance(value, dict):
        print('{}{} {{'.format(indent, key))
        for k, v in value.items():
            dump_pbtxt(k, v, level + 1)
        print('{}}}'.format(indent))


if __name__ == '__main__':
    main()
