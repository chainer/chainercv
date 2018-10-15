cd examples/ssd
curl -L https://cloud.githubusercontent.com/assets/2062128/26187667/9cb236da-3bd5-11e7-8bcf-7dbd4302e2dc.jpg \
     -o sample.jpg

$PYTHON demo.py --model sample.jpg
$PYTHON demo.py --model --gpu 0 sample.jpg
