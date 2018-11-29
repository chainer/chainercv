cd examples/semantic_segmentation
sed -e 's/CamVidDataset(split='\''test'\'')/CamVidDataset(split='\''test'\'').slice[:20]/' -i eval_camvid.py

$PYTHON eval_camvid.py --gpu 0 --batchsize 2
