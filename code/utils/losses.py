import torch
import torch.nn as nn
import numpy as np


def boundary_weighted(loss, boundary_map, boundary_weight=1):
    weights = torch.ones(boundary_map.size()).cuda()
    weights[boundary_map] = boundary_weight
    loss = loss * weights
    return loss.mean()

# 无标记数据熵最小化损失函数
def entropy_loss(v, mask):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()

    loss_image = torch.mul(v, torch.log2(v + 1e-30))
    loss_image = torch.sum(loss_image, dim=1)
    loss_image = mask.float().squeeze(1) * loss_image

    percentage_valid_points = torch.mean(mask.float())

    return -torch.sum(loss_image) / (n * h * w * np.log2(c) * percentage_valid_points)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', boundary_weight=1):
        super(CrossEntropyLoss2d, self).__init__()
        self.boundary_weight = boundary_weight
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction)

    def forward(self, inputs, targets, boundary_map):
        # 计算边界像素的权重值
        weights = torch.ones(targets.size()).cuda()
        weights[boundary_map] = self.boundary_weight

        # 计算交叉熵损失函数并加权
        inputs = torch.log(torch.softmax(inputs, dim=1))    # torch.nn.CrossEntropyLoss相当于softmax + log + NLLLoss
        loss = self.nll_loss(inputs, targets)
        loss = loss * weights
        loss = loss.mean()

        return loss


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes