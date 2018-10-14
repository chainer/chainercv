#! /usr/bin/env sh
set -e

cd examples/ssd

$PYTHON train.py --model ssd300 --batchsize 2 --gpu 0 --short
$PYTHON train.py --model ssd512 --batchsize 2 --gpu 0 --short
