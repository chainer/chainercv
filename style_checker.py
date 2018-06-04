"""Coding style checker for ChainerCV

This is a coding style checker used in ChainerCV.

Usage:
    $ python style_checker.py [--exclude <file> [<file> ...]] <directory>

This script checks the following coding rules.

- Arguments of `ndarray.reshape`.
    If the target shape is 1-d array, it should be passed as an integer.
    Otherwise, it should be passed as a tuple.

    Example:
        a.reshape(3)  # OK
        b.reshape((1, 2, 3))  # OK

        a.reshape((3,))  # NG
        b.reshape(1, 2, 3)  # NG

- Arguments of `ndarray.transpose`.
    The order of axes should be passed as a tuple.

    Example:
        a.transpose((2, 0, 1))  # OK
        a.reshape(2, 0, 1)  # NG

- Initialization of empty `list`/`dict`/`tuple`.
    An empty `list`/`dict`/`tuple` should be initialized by `[]`/`{}`/`()`.

    Example:
        a = []  # OK
        b = {}  # OK
        c = ()  # OK

        a = list()  # NG
        b = dict()  # NG
        c = tuple()  # NG
 """

import argparse
import ast
import os
import sys


def check(source):
    checkers = (
        check_reshape,
        check_transpose,
        check_empty_list,
        check_empty_dict,
    )

    for node in ast.walk(ast.parse(source)):
        for checker in checkers:
            for err in checker(node):
                yield err


def check_reshape(node):
    if not isinstance(node, ast.Call):
        return
    if not isinstance(node.func, ast.Attribute):
        return
    if isinstance(node.func.value, ast.Name) and \
       node.func.value.id in {'np', 'cupy', 'F'}:
        return
    if not node.func.attr == 'reshape':
        return

    if len(node.args) > 1:
        yield (node.lineno, 'reshape(A, B, ...)')

    if len(node.args) == 1 and \
       isinstance(node.args[0], ast.Tuple) and \
       len(node.args[0].elts) == 1:
        yield (node.lineno, 'reshape((A,))')


def check_transpose(node):
    if not isinstance(node, ast.Call):
        return
    if not isinstance(node.func, ast.Attribute):
        return
    if isinstance(node.func.value, ast.Name) and \
       node.func.value.id in {'np', 'cupy', 'F'}:
        return
    if not node.func.attr == 'transpose':
        return

    if len(node.args) > 1:
        yield (node.lineno, 'transpose(A, B, ...)')

    if len(node.args) == 1 and \
       isinstance(node.args[0], ast.Tuple) and \
       len(node.args[0].elts) == 1:
        yield (node.lineno, 'transpose((A,))')


def check_empty_list(node):
    if not isinstance(node, ast.Call):
        return
    if not isinstance(node.func, ast.Name):
        return
    if node.func.id == 'list' and len(node.args) == 0:
        yield (node.lineno, 'init by list()')


def check_empty_dict(node):
    if not isinstance(node, ast.Call):
        return
    if not isinstance(node.func, ast.Name):
        return
    if node.func.id == 'dict' and len(node.args) == 0:
        yield (node.lineno, 'init by dict()')


def check_empty_tuple(node):
    if not isinstance(node, ast.Call):
        return
    if not isinstance(node.func, ast.Name):
        return
    if node.func.id == 'tuple' and len(node.args) == 0:
        yield (node.lineno, 'init by tuple()')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', nargs='+')
    parser.add_argument('dir')
    args = parser.parse_args()

    n_err = 0

    for dir, _, files in os.walk(args.dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if not ext == '.py':
                continue

            if args.exclude is not None and file in args.exclude:
                continue

            path = os.path.join(dir, file)
            lines = open(path).readlines()

            for lineno, msg in check(''.join(lines)):
                print('{:s}:{:d} : {:s}'.format(path, lineno, msg))
                print(lines[lineno - 1])

                n_err += 1

    if n_err > 0:
        sys.exit('{:d} style errors are found.'.format(n_err))


if __name__ == '__main__':
    main()
