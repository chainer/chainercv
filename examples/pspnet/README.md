Pyramid Scene Parsing Network
=============================

# Additional Requirement

- ChainerMN 1.0.0b1+

For training:

- 16 GPUs (each should have 12GB+ memory)

# Preparation

Calculate the mean image of the training set of Cityscapes:

```
python cityscapes.py --img_dir IMD_DIR --label_dir LABEL_DIR --out_dir OUT_DIR
```

where IMG_DIR is a path to "leftImg8bit" dir, LABEL_DIR is a path to "gtFine" or "gtCoarse" dir, and OUT_DIR is a path to the dir where the resulting mean.npy will be saved in.


# Training

Save hostnames you use for training in `hosts.txt`

```
MPLBACKEND=Agg \
mpiexec -prefix /home/shunta/lib/openmpi/install -n 8 -hostfile hosts.txt \
-x PATH -x LD_LIBRARY_PATH -x LIBRARY_PATH -x CPATH -x MPLBACKEND \
python train.py config.yml --gpu --communicator hierarchical
```
