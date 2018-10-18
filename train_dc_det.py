import os
import numpy as np
import argparse
import time

import torch

from utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient, calc_grad_norm
from utils.box_utils import sample_proposals
from model.dc_vgg16 import DC_VGG16_DET, DC_VGG16_CLS
from datasets.tdet_dataset import TDETDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--net', default='DC_VGG16_DET', type=str)
    parser.add_argument('--start_iter', help='starting iteration', default=1, type=int)
    parser.add_argument('--max_iter', help='number of iterations', default=70000, type=int)
    parser.add_argument('--disp_interval', help='number of iterations to display loss', default=1000, type=int)
    parser.add_argument('--save_interval', dest='save_interval', help='number of iterations to save', default=10000, type=int)
    parser.add_argument('--save_dir', help='directory to save models', default="../repo/tdet")
    parser.add_argument('--data_dir', help='directory to load data', default='../data', type=str)

    parser.add_argument('--pooling_method', help='roi_pooling or roi_align', default='roi_pooling', type=str)
    parser.add_argument('--prop_method', help='ss, eb, or mcg', default='eb', type=str)
    parser.add_argument('--prop_min_scale', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--num_prop', help='maximum number of proposals to use for training', default=2000, type=int)
    parser.add_argument('--bs', help='training batch size', default=128, type=int)
    parser.add_argument('--pos_ratio', help='ratio of positive roi', default=0.25, type=float)

    parser.add_argument('--lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--s', dest='session', help='training session', default=0, type=int)
    parser.add_argument('--seed', help='random sed', default=1, type=int)

    parser.add_argument('--target_only', action='store_true')
    parser.add_argument('--pretrained_base_path', type=str)
    args = parser.parse_args()
    return args


def draw_box(boxes, col=None):
    for j, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if col is None:
            c = np.random.rand(3)
        else:
            c = col
        plt.hlines(ymin, xmin, xmax, colors=c, lw=2)
        plt.hlines(ymax, xmin, xmax, colors=c, lw=2)
        plt.vlines(xmin, ymin, ymax, colors=c, lw=2)
        plt.vlines(xmax, ymin, ymax, colors=c, lw=2)


def validate(model, val_dataset, args, device):
    model.eval()
    tot_loss = 0
    for step in range(len(val_dataset)):
        batch = val_dataset.get_data(step, h_flip=False, target_im_size=688)

        im_data = batch['im_data'].unsqueeze(0).to(device)
        proposals = batch['proposals']
        gt_boxes = batch['gt_boxes']
        gt_labels = batch['gt_labels']
        pos_cls = [i for i in range(20) if i in gt_labels]

        loss = 0
        for cls in np.random.choice(pos_cls, 2):
            indices = np.where(gt_labels.numpy() == cls)[0]
            here_gt_boxes = gt_boxes[indices]
            here_proposals, here_labels, _, pos_cnt, neg_cnt = sample_proposals(here_gt_boxes, proposals, args.bs // 2, args.pos_ratio)
            here_proposals = here_proposals.to(device)
            here_labels = here_labels.to(device)
            here_loss = model(im_data, cls, here_proposals, here_labels)
            loss = loss + here_loss.item()
        loss /= 2
        tot_loss += loss
    model.train()
    print('Validation loss: %.4f' % (tot_loss / len(val_dataset)))


def train():
    args = parse_args()
    print('Called with args:')
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.target_only:
        source_train_dataset = TDETDataset(['voc07_trainval'], args.data_dir, args.prop_method,
                                       num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)
    else:
        source_train_dataset = TDETDataset(['coco60_train2014', 'coco60_val2014'], args.data_dir, args.prop_method,
                                           num_classes=60, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)
    target_val_dataset = TDETDataset(['voc07_test'], args.data_dir, args.prop_method,
                                       num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

    lr = args.lr

    if args.net == 'DC_VGG16_DET':
        base_model = DC_VGG16_CLS(None, 20 if args.target_only else 80, 3, 4)
        checkpoint = torch.load(args.pretrained_base_path)
        base_model.load_state_dict(checkpoint['model'])
        del checkpoint
        model = DC_VGG16_DET(base_model, args.pooling_method)

    optimizer = model.get_optimizer(args.lr)

    log_file_name = os.path.join(output_dir, 'log_{}_{}.txt'.format(args.net, args.session))
    log_file = open(log_file_name, 'w')
    log_file.write(str(args))
    log_file.write('\n')

    model.to(device)
    model.train()
    source_loss_sum = 0
    source_pos_prop_sum = 0
    source_neg_prop_sum = 0
    start = time.time()
    optimizer.zero_grad()
    for step in range(args.start_iter, args.max_iter + 1):
        if step % len(source_train_dataset) == 1:
            source_rand_perm = np.random.permutation(len(source_train_dataset))

        source_index = source_rand_perm[step % len(source_train_dataset)]

        source_batch = source_train_dataset.get_data(source_index, h_flip=np.random.rand() > 0.5, target_im_size=np.random.choice([480, 576, 688, 864, 1200]))

        source_im_data = source_batch['im_data'].unsqueeze(0).to(device)
        source_proposals = source_batch['proposals']
        source_gt_boxes = source_batch['gt_boxes']
        if args.target_only:
            source_gt_labels = source_batch['gt_labels']
        else:
            source_gt_labels = source_batch['gt_labels'] + 20
        source_pos_cls = [i for i in range(80) if i in source_gt_labels]

        source_loss = 0
        for cls in np.random.choice(source_pos_cls, 2):
            indices = np.where(source_gt_labels.numpy() == cls)[0]
            here_gt_boxes = source_gt_boxes[indices]
            here_proposals, here_labels, _, pos_cnt, neg_cnt = sample_proposals(here_gt_boxes, source_proposals, args.bs // 2, args.pos_ratio)
            # plt.imshow(source_batch['raw_img'])
            # draw_box(here_proposals[:pos_cnt] / source_batch['im_scale'], 'black')
            # draw_box(here_proposals[pos_cnt:] / source_batch['im_scale'], 'yellow')
            # plt.show()
            here_proposals = here_proposals.to(device)
            here_labels = here_labels.to(device)
            here_loss = model(source_im_data, cls, here_proposals, here_labels)
            source_loss = source_loss + here_loss

            source_pos_prop_sum += pos_cnt
            source_neg_prop_sum += neg_cnt

        source_loss = source_loss / 2

        source_loss_sum += source_loss.item()
        source_loss.backward()

        clip_gradient(model, 10.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % args.disp_interval == 0:
            end = time.time()
            source_loss_sum /= args.disp_interval
            source_pos_prop_sum /= args.disp_interval
            source_neg_prop_sum /= args.disp_interval
            log_message = "[%s][session %d][iter %4d] loss: %.4f, pos_prop: %.1f, neg_prop: %.1f, lr: %.2e, time: %.1f" % \
                          (args.net, args.session, step, source_loss_sum, source_pos_prop_sum, source_neg_prop_sum, lr, end - start)
            print(log_message)
            log_file.write(log_message + '\n')
            log_file.flush()
            source_loss_sum = 0
            source_pos_prop_sum = 0
            source_neg_prop_sum = 0
            start = time.time()

        if step in (args.max_iter * 4 // 7, args.max_iter * 6 // 7):
            adjust_learning_rate(optimizer, 0.1)
            lr *= 0.1

        if step % args.save_interval == 0 or step == args.max_iter:
            validate(model, target_val_dataset, args, device)
            save_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.session, step))
            checkpoint = dict()
            checkpoint['net'] = args.net
            checkpoint['session'] = args.session
            checkpoint['pooling_method'] = args.pooling_method
            checkpoint['iterations'] = step
            checkpoint['model'] = model.state_dict()

            save_checkpoint(checkpoint, save_name)
            print('save model: {}'.format(save_name))

    log_file.close()


if __name__ == '__main__':
    train()