import argparse
import logging
import os
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.metal_datasets import (LabeledDataset, UnlabeledDataset, ValDataset, RandomGenerator, Resize)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.metric_for_metal import get_confusion_metrix, pixelAccuracy, IoU, MIoU
from utils.memory_bank import MemoryBank


def get_current_weight(epoch, lamda):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return lamda * ramps.sigmoid_rampup(epoch, args.rampup_epochs)


def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen


def filter_sharp(origin, sharpen_map):
    sharpen = sharpening(origin)
    return torch.where(sharpen_map, sharpen, origin)


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str, default='UHCS', help='Name of dataset')
parser.add_argument('--model', type=str, default='con2Net_v2', help='model_name')
parser.add_argument('--deterministic', type=bool,  default=True, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=1221, help='random seed')
parser.add_argument('--batch_size', type=int,  default=8, help='total batch size, >=2')
# labeled and unlabeled
parser.add_argument('--labeled_proportion', type=float, default=0.5, help='proportion of labeled data in training set')
# costs weight
parser.add_argument('--rampup_epochs', type=float, default=200.0, help='rampup_epochs')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda_seg', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--lamda_consist', type=float, default=0.1, help='weight contrast loss')
parser.add_argument('--lamda_contrast', type=float, default=0.1, help='weight of feature consistency loss')
parser.add_argument('--lamda_pseudo', type=float, default=0.1, help='weight of net consistency loss')
# num_workers
parser.add_argument('--num_workers', type=int, default=0, help='num_workers for training set')
# contrast
parser.add_argument('--feature_length', type=int, default=256, help='dims of contrastive feature vector')
parser.add_argument('--queue_size', type=int, default=256, help='length of queue')
parser.add_argument('--contrast_T', type=float, default=0.1, help='temperature of infoNCE loss')
parser.add_argument('--top_k', type=int, default=4, help='top k')
parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold of pixel filtering')
# sharpen_scope
parser.add_argument('--sharpen_scope', type=str, default="only_accurate", help='scope of sharping')


args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.batch_size // 2
    unlabeled_bs = args.batch_size - labeled_bs
    confidence = args.confidence

    if args.dataset_name == 'UHCS':
        max_iters = 2000
        val_per_iter = 10
        weight_vary_iter = 10
        num_classes = 4
        patch_size = (320, 320)
        val_bs = 4
    else:
        max_iters = 2000
        val_per_iter = 10
        weight_vary_iter = 10
        num_classes = 5
        patch_size = (320, 320)
        val_bs = 1

    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes, feature_length=args.feature_length)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    memory_bank = MemoryBank(
        num_classes=num_classes,
        queue_size=args.queue_size,
        temperature=args.contrast_T,
        top_k=args.top_k
    )

    labeled_dataset = LabeledDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        labeled_proportion=args.labeled_proportion,
        transform=RandomGenerator(patch_size, data_type='labeled')
    )

    unlabeled_dataset = UnlabeledDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        transform=RandomGenerator(patch_size, data_type='unlabeled')
    )

    val_dataset = ValDataset(
        base_dir=args.root_path,
        dataset_name=args.dataset_name,
        transform=Resize(patch_size)
    )

    total_img = len(labeled_dataset) + len(unlabeled_dataset)
    print("Total images: {}, labeled: {}, unlabeled: {}, val: {}".format(total_img, len(labeled_dataset), len(unlabeled_dataset), len(val_dataset)))

    labeled_loader = DataLoader(labeled_dataset, batch_size=labeled_bs, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=unlabeled_bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=0)

    unlabeled_iter = enumerate(unlabeled_loader)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))

    # loss function
    ce_loss = CrossEntropyLoss()
    pseudo_ce_loss = CrossEntropyLoss()
    consistency_criterion = losses.mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(labeled_loader)))
    #
    best_miou = 0.0
    best_acc = 0.0
    best_iou = []
    best_iter = 0

    item_num = 0
    max_epoch = max_iters // len(labeled_loader) + 1

    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        # train
        for _, labeled_data in enumerate(labeled_loader):   # every epoch
            try:
                _, unlabeled_data = unlabeled_iter.__next__()
            except:
                unlabeled_iter = enumerate(unlabeled_loader)
                _, unlabeled_data = unlabeled_iter.__next__()

            labeled_batch, gt_batch, unlabeled_batch = labeled_data['image'], labeled_data['label'], unlabeled_data['image']
            image_batch = torch.cat([labeled_batch, unlabeled_batch], dim=0)
            image_batch, gt_batch = image_batch.cuda(), gt_batch.cuda()

            model.train()

            all_outputs = model(image_batch)
            outputs, features = all_outputs['outputs'], all_outputs['features']

            num_outputs = len(outputs)

            y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
            y_sharpen_label = torch.zeros((num_outputs,) + outputs[0].shape)
            # only for labeled
            item_num = item_num + 1
            loss_seg = 0
            loss_pseudo = 0
            loss_contrast = 0

            for idx in range(num_outputs):
                # labeled
                y = outputs[idx][:len(gt_batch), ...]
                l_feature = features[idx][:len(gt_batch), ...]
                y_prob = F.softmax(y, dim=1)
                y_confidence_map, y_hard_pred = torch.max(y_prob, dim=1)

                # filter accurate pixel feature from labeled pixel, and get Corresponding projection feature
                l_acc_map = torch.eq(y_hard_pred, gt_batch) & (y_confidence_map > confidence)
                l_acc_confidence = y_confidence_map[l_acc_map]
                l_acc_label = gt_batch[l_acc_map]
                l_acc_feature = l_feature.permute(0, 2, 3, 1)[l_acc_map, ...]
                l_contrast_feature = model.forward_projection_head(l_acc_feature)

                # memory bank update
                loss_contrast += memory_bank.in_queue_get_loss(confidence=l_acc_confidence, label=l_acc_label, contrast_feature=l_contrast_feature)

                # unlabeled pseudo_label
                with torch.no_grad():
                    unl_pred = outputs[idx][len(gt_batch):, ...]
                    unl_hard_pred = torch.argmax(torch.softmax(unl_pred, dim=1), dim=1)
                    unl_feature = features[idx][len(gt_batch):, ...].permute(0, 2, 3, 1)
                    f_new_shape = unl_feature.shape[0]*unl_feature.shape[1]*unl_feature.shape[2], unl_feature.shape[3]
                    cf_new_shape = unl_feature.shape[0], unl_feature.shape[1], unl_feature.shape[2], args.feature_length
                    unl_contrast_feature = model.forward_projection_head(unl_feature.reshape(f_new_shape)).reshape(cf_new_shape)

                # get unlabeled sharp map through contrast feature with memory bank
                pseudo_label = memory_bank.get_pseudo_label(unl_contrast_feature)
                unl_acc_map = torch.eq(unl_hard_pred, pseudo_label)

                # calculate segment loss and contrast loss
                loss_seg += ce_loss(y, gt_batch.long())
                if memory_bank.flag:
                    loss_pseudo += pseudo_ce_loss(unl_pred, pseudo_label.long())

                # get sharpen map
                l_acc_map = torch.eq(y_hard_pred, gt_batch).unsqueeze(1).repeat(1, num_classes, 1, 1)
                unl_acc_map = unl_acc_map.unsqueeze(1).repeat(1, num_classes, 1, 1)
                sharpen_map = torch.cat([l_acc_map, unl_acc_map], dim=0)
                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all     # 保存labeled和unlabeled输出softmax后的结果
                if args.sharpen_scope == 'only_accurate':
                    y_sharpen_label[idx] = filter_sharp(y_prob_all, sharpen_map)   # 保存labeled和unlabeled输出softmax并锐化后的结果
                elif args.sharpen_scope == 'all':
                    y_sharpen_label[idx] = sharpening(y_prob_all)
                elif args.sharpen_scope == 'none':
                    y_sharpen_label[idx] = y_prob_all

            # consistency loss
            loss_consist = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_consist += consistency_criterion(y_sharpen_label[i], y_ori[j])

            # loss weight
            contrast_weight = get_current_weight(item_num//weight_vary_iter, args.lamda_contrast)
            consist_weight = get_current_weight(item_num//weight_vary_iter, args.lamda_consist)
            pseudo_weight = get_current_weight(item_num//weight_vary_iter, args.lamda_pseudo)
            # total loss
            loss = args.lamda_seg*loss_seg + consist_weight*loss_consist + contrast_weight*loss_contrast + pseudo_weight*loss_pseudo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info('iter %d : loss : %03f, loss_seg: %03f, loss_contrast: %03f, loss_consist: %03f, loss_pseudo: %03f'
                         % (item_num, loss, loss_seg, loss_contrast, loss_consist, loss_pseudo))

            # valuate
            if item_num > 0 and item_num % val_per_iter == 0:
                writer.add_scalars('train', {'loss': loss}, item_num)
                writer.add_scalars('train', {'loss_seg': loss_seg}, item_num)
                writer.add_scalars('train', {'loss_contrast': loss_contrast}, item_num)
                writer.add_scalars('train', {'loss_consist': loss_consist}, item_num)
                writer.add_scalars('train', {'loss_pseudo': loss_pseudo}, item_num)

                model.eval()

                confusion_metrix = 0
                val_seg_loss = 0.0

                for _, sampled_batch in enumerate(valloader):
                    val_image_batch, val_gt_batch, val_origin_gt = sampled_batch['image'], sampled_batch['label'], sampled_batch['origin_gt']
                    val_image_batch, val_gt_batch, val_origin_gt = val_image_batch.cuda(), val_gt_batch.cuda(), val_origin_gt.cuda()

                    val_output = model(val_image_batch, mode='val')
                    val_seg_loss += ce_loss(val_output, val_gt_batch.long())

                    confusion_metrix += get_confusion_metrix(val_output, val_origin_gt, torch.device('cuda'), num_classes=num_classes)

                acc = pixelAccuracy(confusion_metrix)
                iou = IoU(confusion_metrix)
                miou = MIoU(iou)

                writer.add_scalar('val_loss', val_seg_loss, item_num)
                writer.add_scalars('metric', {'acc': acc}, item_num)
                writer.add_scalars('metric', {'miou': miou}, item_num)

                # record best
                if miou > best_miou:
                    best_miou = miou
                    best_acc = acc
                    best_iou = iou
                    best_iter = item_num
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                    with open(snapshot_path + '/best_record.txt', mode='w', encoding='utf-8') as f:
                        f.write("best_iter_num : %d, train_seg_loss: %03f ,val_seg_loss: %03f, acc: %03f, miou: %03f" % (best_iter, loss_seg, val_seg_loss, best_acc, best_miou))
                        f.write("\nIoU:" + str(best_iou))

                logging.info('iter %d : miou : %f  acc : %f' % (item_num, miou, acc))
                model.train()

            if item_num >= max_iters:
                break
        if item_num >= max_iters:
            iterator.close()
            break
            
    writer.close()

    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    param_folder_name = "contrast-{}_consist-{}_pseudo-{}".format(args.lamda_contrast, args.lamda_consist, args.lamda_pseudo)
    proportion_folder_name = "{}-labeled".format(args.labeled_proportion)

    snapshot_path = "../result/{}/{}/{}/{}".format(args.dataset_name, args.model, param_folder_name, proportion_folder_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # train
    train(args, snapshot_path)


