#!/bin/zsh
python test.py \
    --device 0 \
    --data data/trash.yaml \
    --img-size 1024 \
    --weights '/opt/ml/level2/baseline/yolov7/runs/train/exp10/weights/best.pt' \
    --name 1130_kh_yolov7-e6e \
    --project Detection-Competition \
    --task test \
    --save-json