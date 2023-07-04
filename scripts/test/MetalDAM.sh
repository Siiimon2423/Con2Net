#!/bin/bash
python ../../code/test.py  --dataset_name MetalDAM  --model con2Net_v2  --labeled_proportion 1.0 && \
python ../../code/test.py  --dataset_name MetalDAM  --model con2Net_v2  --labeled_proportion 0.5 && \
python ../../code/test.py  --dataset_name MetalDAM  --model con2Net_v2  --labeled_proportion 0.25
