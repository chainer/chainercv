# Classification

## ImageNet

### Weight conversion

Single crop error rates of the models with the weights converted from Caffe weights.

| Model | Top 1 | Original Top 1 |
|:-:|:-:|:-:|
| MobileNetV2-1.0 | 28.3 % | 28.0 % [6] |
| MobileNetV2-1.4 | 24.3 % | 25.3 % [6] |
| VGG16 | 29.0 % | 28.5 % [1] |
| ResNet50 (`arch=he`) | 24.8 % | 24.7 % [2] |
| ResNet101 (`arch=he`) | 23.6 % | 23.6 % [2] |
| ResNet152 (`arch=he`) | 23.2 % | 23.0 % [2] |
| SE-ResNet50 | 22.7 % | 22.4 % [3,4] |
| SE-ResNet101 | 21.8 % | 21.8 % [3,4] |
| SE-ResNet152 | 21.4 % | 21.3 % [3,4] |
| SE-ResNeXt50 | 20.9 % | 21.0 % [3,4] |
| SE-ResNeXt101 | 19.7 % | 19.8 % [3,4] |

Ten crop error rate.

| Model | Top 1 | Original Top 1 |
|:-:|:-:|:-:|
| MobileNetV2-1.0 | 25.6 % |  |
| MobileNetV2-1.4 | 22.4 % |  |
| VGG16 | 27.1 % |   |
| ResNet50 (`arch=he`) | 23.0 % | 22.9 % [2] |
| ResNet101 (`arch=he`) | 21.8 % | 21.8 % [2] |
| ResNet152 (`arch=he`) | 21.4 % | 21.4 % [2] |
| SE-ResNet50 | 20.8 % |  |
| SE-ResNet101 | 20.1 % |  |
| SE-ResNet152 | 19.7 % |  |
| SE-ResNeXt50 | 19.4 % |  |
| SE-ResNeXt101 | 18.6 % |  |


The results can be reproduced by the following command.
These scores are obtained using OpenCV backend. If Pillow is used, scores would differ.

```
$ python eval_imagenet.py <path_to_val_dataset> [--model mobilenet_v2|vgg16|resnet50|resnet101|resnet152|se-resnet50|se-resnet101|se-resnet152] [--pretrained-model <model_path>] [--batchsize <batchsize>] [--gpu <gpu>] [--crop center|10]
```

### Trained model

Single crop error rates of the models trained with the ChainerCV's training script.

| Model | Top 1 | Original Top 1 |
|:-:|:-:|:-:|
| ResNet50 (`arch=fb`) | 23.51 % | 23.60% [5] |
| ResNet101 (`arch=fb`) | 22.07 % | 22.08% [5] |
| ResNet152 (`arch=fb`) | 21.67 % |  |


The scores of the models trained with `train_imagenet_multi.py`, which can be executed like below.
Please consult the full list of arguments for the training script with `python train_imagenet_multi.py -h`.
```
$ mpiexec -n N python train_imagenet_multi.py <path_to_train_dataset> <path_to_val_dataset>
```

The training procedure carefully follows the "ResNet in 1 hour" paper [5].

#### Performance tip
cuDNN convolution functions can be optimized with extra commands (see https://docs.chainer.org/en/stable/performance.html#optimize-cudnn-convolution).

#### Detailed training results

Here, we investigate the effect of the number of GPUs on the final performance.
For more statistically reliable results, we obtained results from five different random seeds.

| Model | # GPUs | Top 1 |
|:-:|:-:|:-:|
| ResNet50 (`arch=fb`) | 8 | 23.53 (std=0.06) |
| ResNet50 (`arch=fb`) | 32 | 23.56 (std=0.11) |


## How to prepare ImageNet dataset

This instructions are based on the instruction found [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  $ mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  $ tar -xvf ILSVRC2012_img_train.tar && mv ILSVRC2012_img_train.tar ..
  $ find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  $ cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  $ mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  $ wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  $ mv ILSVRC2012_img_val.tar .. && cd ..
  ```


## References

1. Karen Simonyan, Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition" ICLR 2015
2. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition" CVPR 2016
3. Jie Hu, Li Shen, Gang Sun. "Squeeze-and-Excitation Networks" CVPR 2018
4. https://github.com/hujie-frank/SENet
5. Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" https://arxiv.org/abs/1706.02677
6. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" https://arxiv.org/abs/1801.04381
