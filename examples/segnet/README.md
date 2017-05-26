SegNet
======

# Preparation

Please create `class_weight.npy` using calc_weight.py first. Just run:

```
python calc_weight.py
```

# Start training

```
CHAINER_SEED=0 CHAINER_TYPE_CHECK=0 python -OO -W ignore train.py --gpu 0
```

# Evaluation

```
bash evaluate.sh [GPU ID] [MODEL SNAPSHOT]
```

e.g.,

```
bash evaluate.sh 0 result/result/2017-05-26_23-01-06/model_iteration-10000
```

You will see the values of intersection over union (IoU) for each class,
the mean IoU over all the classes, class average accuracy, and global average
accuracy like this:

```
                    Sky : 0.8525
               Building : 0.6406
                   Pole : 0.1531
           Road_marking : 0.8231
                   Road : 0.5541
               Pavement : 0.5396
                   Tree : 0.1528
             SignSymbol : 0.1899
              Fence,Car : 0.5955
             Pedestrian : 0.2392
              Bicyclist : 0.1846
================================
               mean IoU : 0.4477
 Class average accuracy : 0.6456
Global average accuracy : 0.8059
```
