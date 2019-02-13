cd examples/detection
sed -e 's/return_crowded=True)/return_crowded=True).slice[:20]/' -i eval_coco_multi.py

$MPIEXEC $PYTHON eval_coco_multi.py --model faster_rcnn_fpn_resnet50 --batchsize 2
$MPIEXEC $PYTHON eval_coco_multi.py --model faster_rcnn_fpn_resnet101 --batchsize 2
