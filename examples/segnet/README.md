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
0: 0.885903675261
1: 0.694224707893
2: 0.182279186454
3: 0.77437463848
4: 0.353746855831
5: 0.649355357139
6: 0.181990664416
7: 0.162778639792
8: 0.595895743803
9: 0.314330561759
10: 0.315552359398
mean: 0.464584762748
```
