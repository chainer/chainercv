cd examples/instance_segmentation
sed -e 's/return_crowded=True)/return_crowded=True).slice[:20]/' -i eval_instance_segmentation.py

$PYTHON eval_instance_segmentation.py --dataset coco --model mask_rcnn_fpn_resnet50 --batchsize 2 --gpu 0
