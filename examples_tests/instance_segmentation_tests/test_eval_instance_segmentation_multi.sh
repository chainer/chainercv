cd examples/instance_segmentation
sed -e 's/return_crowded=True)/return_crowded=True).slice[:20]/' -i eval_instance_segmentation.py

$MPIEXEC $PYTHON eval_instance_segmentation_multi.py --dataset coco --model mask_rcnn_fpn_resnet50 --batchsize 2
