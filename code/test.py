import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.net_factory import net_factory
from utils.metric_for_metal import get_confusion_metrix, pixelAccuracy, IoU, MIoU
from dataloaders.metal_datasets import (LabeledDataset, UnlabeledDataset, ValDataset, RandomGenerator, Resize)
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--snapshot_path', type=str, default='../pth', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str, default='MetalDAM', help='Name of dataset')
parser.add_argument('--model', type=str, default='con2Net_v2', help='model_name')
parser.add_argument('--labeled_proportion', type=float, default=1.0, help='proportion of labeled data in training set')
parser.add_argument('--feature_length', type=int, default=256, help='')

args = parser.parse_args()


def save_predict_img(preds, origin_size, dir_path, name):

    if args.dataset_name == 'MetalDAM':
        color_map = np.array([
            (255, 100, 21),  # 0 蓝  1Matrix 铁素体基体
            (0, 255, 255),  # 1 黄  Austenite 奥氏体
            (40, 40, 254),  # 2 红  Martensite/Austenite (MA) 马氏体/奥氏体（MA）
            (40, 176, 75),  # 3 绿  Precipitate 沉淀物
            (0,   0,   0)   # 4 黑  Defect 缺陷
        ], np.uint8)
    else:
        color_map = np.array([
            (255, 100, 21),  # 0 蓝   Ferritic matrix 铁素体基体
            (0, 255, 255),  # 1 黄   Cementite network 渗碳体网络
            (40, 40, 254),  # 2 红   Spheroidite particles 球化体颗粒
            (40, 176, 75),  # 3 绿   Widmanstätten laths 魏氏体条纹
        ], np.uint8)

    upper = nn.UpsamplingBilinear2d(size=origin_size)
    preds = upper(preds)
    preds = torch.argmax(preds[0], dim=0).cpu().numpy().astype(np.uint8)
    color_out = color_map[preds]
    cv2.imwrite('{}/{}.png'.format(dir_path, args.model+'-'+name), color_out)


if __name__ == "__main__":

    snapshot_path = args.snapshot_path + '/{}'.format(args.dataset_name)
    img_save_path = os.path.join(snapshot_path, "{}_pred_images".format(args.model))
    if not os.path.isdir(img_save_path):
        os.makedirs(img_save_path)

    if args.dataset_name == 'UHCS':
        num_classes = 4
        patch_size = (320, 320)
    else:
        num_classes = 5
        patch_size = (320, 320)

    static_path = snapshot_path+'/{}_best_model_{}.pth'.format(args.model, args.labeled_proportion)
    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes, feature_length=args.feature_length)
    model.load_state_dict(torch.load(static_path), strict=False)
    model.eval()

    val_dataset = ValDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        transform=Resize(patch_size)
    )
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    confusion_metrix = 0
    for _, sampled_batch in enumerate(valloader):
        val_image_batch, val_gt_batch, val_origin_gt, img_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['origin_gt'],  sampled_batch['name'][0]
        val_image_batch, val_gt_batch, val_origin_gt = val_image_batch.cuda(), val_gt_batch.cuda(), val_origin_gt.cuda()
        if args.model == 'con2Net_v2':
            val_output = model(val_image_batch, mode='val')
        else:
            val_output = model(val_image_batch)[0]

        confusion_metrix += get_confusion_metrix(val_output, val_origin_gt, torch.device('cuda'),
                                                 num_classes=num_classes)
        save_predict_img(val_output, (val_origin_gt.shape[1], val_origin_gt.shape[2]), img_save_path, img_name)

    acc = pixelAccuracy(confusion_metrix)
    iou = IoU(confusion_metrix)
    miou = MIoU(iou)

    with open(snapshot_path + '/{}_best_test_{}.txt'.format(args.model, args.labeled_proportion), mode='w', encoding='utf-8') as f:
        f.write("acc: %03f, miou: %03f" % (acc, miou))
        f.write("\nIoU:" + str(iou))

