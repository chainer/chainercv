cd examples/detection
sed -e 's/return_crowded=True)/return_crowded=True).slice[:20]/' -i eval_coco.py

$PYTHON eval_coco.py --model faster_rcnn_fpn_resnet50 --gpu 0 --batchsize 2
$PYTHON eval_coco.py --model faster_rcnn_fpn_resnet101 --gpu 0 --batchsize 2
