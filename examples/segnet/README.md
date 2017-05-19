SegNet
======

# Preparation

Please create `class_weight.npy` using calc_weight.py first. Just run:

```
python calc_weight.py
```

# Start training

```
CHAINER_TYPE_CHECK=0 python -OO -W ignore train.py --gpu 0
```
