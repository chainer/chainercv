cd examples/detection
sed -e 's/return_difficult=True)/return_difficult=True).slice[:20]/' -i eval_voc07.py

$PYTHON eval_voc07.py --model faster_rcnn --gpu 0 --batchsize 2
$PYTHON eval_voc07.py --model ssd300 --gpu 0 --batchsize 2
$PYTHON eval_voc07.py --model ssd512 --gpu 0 --batchsize 2
$PYTHON eval_voc07.py --model yolo_v2 --gpu 0 --batchsize 2
$PYTHON eval_voc07.py --model yolo_v3 --gpu 0 --batchsize 2
