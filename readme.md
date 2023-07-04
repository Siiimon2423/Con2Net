# Con2Net

### Introduction
This repository is for the paper:
'Con2Net: A Semi-Supervised Model Based on Consistency Regularization and Contrastive Learning for Metallographic Image Segmentation'. 

### Requirements
* numpy==1.21.5
* opencv_python==4.6.0.66
* tensorboardX==2.6.1
* torch==1.11.0
* torchmetrics==0.10.3
* torchvision==0.12.0
* tqdm==4.64.0

All experiments in our paper were conducted on a single NVIDIA Tesla V100 GPU.

### Usage
1. Install 
```
pip install -r requirements.txt
```
2. Train the model
```
cd ./scripts/con2net
# on MetalDAM dataset
bash MetalDAM.sh
# on UHCS dataset
bash UHCS.sh
```
3. Test the model
```
cd ./scripts/test
bash MetalDAM.sh
bash UHCS.sh
```
