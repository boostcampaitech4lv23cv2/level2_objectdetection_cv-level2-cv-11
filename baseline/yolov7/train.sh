#!/bin/zsh
python train_aux.py \
    --workers 8 \
    --device 0 \
    --epochs 300 \
    --batch-size 8 \
    --weights 'weights/pseudo-3.pt' \
    --data data/trash-full.yaml \
    --img-size 1024 1024 \
    --cfg cfg/training/yolov7-e6e.yaml \
    --hyp data/my-hyp.yaml \
    --name 1201_kh_yolov7-e6e-train-full \
    --entity miho \
    --project Detection-Competition


# python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
