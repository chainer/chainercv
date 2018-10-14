#! /usr/bin/env sh
set -e

cd examples/ssd

$PYTHON train.py --model ssd300 --gpu 0 --short
$PYTHON train.py --model ssd512 --gpu 0 --short
