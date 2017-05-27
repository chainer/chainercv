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
bash evaluate.sh 0 result/2017-05-27_02-18-07/model_iteration-16000
```

You will see the values of intersection over union (IoU) for each class,
the mean IoU over all the classes, class average accuracy, and global average
accuracy like this:

```
                    Sky : 0.8675
               Building : 0.6487
                   Pole : 0.1795
           Road_marking : 0.8493
                   Road : 0.6230
               Pavement : 0.6051
                   Tree : 0.1813
             SignSymbol : 0.1901
              Fence,Car : 0.5924
             Pedestrian : 0.2402
              Bicyclist : 0.2111
==================================
               mean IoU : 0.4716
 Class average accuracy : 0.6705
Global average accuracy : 0.8266
```
