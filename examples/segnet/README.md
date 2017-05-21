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

# Evaluation

```
python evaluate.py --gpu 0 --snapshot results/2017-05-01_09-00-00/model_iteratioin_9000
```

You will see the values of mean intersection over union (mIoU) for each class and the average mIoU:

```
0: 0.879126693456
1: 0.703081561862
2: 0.198033822079
3: 0.8427641436
4: 0.607411863769
5: 0.628452506525
6: 0.209502780697
7: 0.241587759845
8: 0.615301416933
9: 0.326346484168
10: 0.302901995467
mean: 0.504955548036
```
