#!/bin/bash
python ../../code/train_con2Net.py \
  --dataset_name UHCS \
  --batch_size 8 \
  --model con2Net_v2 \
  --lamda_contrast 0.1 \
  --lamda_consist 0.1 \
  --lamda_pseudo 0.1 \
  --labeled_proportion 1.0 \
  --confidence 0.8 && \
python ../../code/train_con2Net.py \
  --dataset_name UHCS \
  --batch_size 8 \
  --model con2Net_v2 \
  --lamda_contrast 0.1 \
  --lamda_consist 0.1 \
  --lamda_pseudo 0.1 \
  --labeled_proportion 0.5 \
  --confidence 0.8 && \
python ../../code/train_con2Net.py \
  --dataset_name UHCS \
  --batch_size 8 \
  --model con2Net_v2 \
  --lamda_contrast 0.1 \
  --lamda_consist 0.1 \
  --lamda_pseudo 0.1 \
  --labeled_proportion 0.25 \
  --confidence 0.8