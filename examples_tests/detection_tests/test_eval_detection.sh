cd examples/detection
sed -e 's/return_difficult=True)/return_difficult=True).slice[:20]/' -i eval_detection.py
sed -e 's/return_crowded=True)/return_crowded=True).slice[:20]/' -i eval_detection.py

$PYTHON eval_detection.py --dataset voc --model ssd300 --batchsize 2 --gpu 0
$PYTHON eval_detection.py --dataset coco --model faster_rcnn_fpn_resnet50 --batchsize 2 --gpu 0
