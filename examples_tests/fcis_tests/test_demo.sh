cd examples/fcis
curl -L https://cloud.githubusercontent.com/assets/2062128/26187667/9cb236da-3bd5-11e7-8bcf-7dbd4302e2dc.jpg \
     -o sample.jpg

# CPU test is too slow
# $PYTHON demo.py --dataset sbd sample.jpg
$PYTHON demo.py --dataset sbd --gpu 0 sample.jpg
# CPU test is too slow
# $PYTHON demo.py --dataset coco sample.jpg
$PYTHON demo.py --dataset coco --gpu 0 sample.jpg
