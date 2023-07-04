#!/bin/bash
python ../../code/train_baseline.py  --dataset_name UHCS  --batch_size 8  --model mcnet2d_v2  --labeled_proportion 1.0 && \
python ../../code/train_baseline.py  --dataset_name UHCS  --batch_size 8  --model mcnet2d_v2  --labeled_proportion 0.5 && \
python ../../code/train_baseline.py  --dataset_name UHCS  --batch_size 8  --model mcnet2d_v2  --labeled_proportion 0.25
