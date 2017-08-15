import argparse
import ast
import os
import sys


def check(source):
    checkers = (
        check_reshape,
        check_transpose,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()

    n_err = 0

    for dir, _, files in os.walk(args.dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if not ext == '.py':
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
