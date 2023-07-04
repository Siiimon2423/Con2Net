import numpy as np
import torch
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix
from torch import nn


def get_confusion_metrix(preds, targets, device, num_classes):

    size = (targets.shape[1], targets.shape[2])
    upper = nn.UpsamplingBilinear2d(size=size)
    preds = upper(preds)

    if num_classes == 5:
        metric = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=3).to(device)   # MetalDAM
    else:
        metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)   # UHCS

    confusion_matrix = metric(preds, targets).cpu().numpy()

    return confusion_matrix


def pixelAccuracy(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


def IoU(confusion_matrix):
    intersection = np.diag(confusion_matrix)  # 取对角元素的值，返回列表
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
        confusion_matrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
    IoU = intersection / union  # 返回列表，其值为各个类别的IoU
    return IoU


def MIoU(IoU):
    return np.nanmean(IoU)

