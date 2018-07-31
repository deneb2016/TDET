import os
import numpy as np
import argparse
import time

import torch

from utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient, calc_grad_norm
from utils.box_utils import sample_proposals
from model.tdet_vgg16 import TDET_VGG16
from datasets.tdet_dataset import TDETDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--net', default='TDET_VGG16', type=str)
    parser.add_argument('--start_iter', help='starting iteration', default=1, type=int)
    parser.add_argument('--max_iter', help='number of iterations', default=70000, type=int)
    parser.add_argument('--disp_interval', help='number of iterations to display loss', default=1000, type=int)
    parser.add_argument('--save_interval', dest='save_interval', help='number of iterations to save', default=10000, type=int)
    parser.add_argument('--save_dir', help='directory to save models', default="../repo/tdet")
    parser.add_argument('--data_dir', help='directory to load data', default='../data', type=str)

    parser.add_argument('--pooling_method', help='roi_pooling or roi_align', default='roi_pooling', type=str)
    parser.add_argument('--cls_specific', help='avg, ind, or no', type=str, default='no')
    parser.add_argument('--backprop2det', help='whether to backprop from score to det', action='store_true')
    parser.add_argument('--det_choice', help='whether to use det choice layer', type=int, default=1)
    parser.add_argument('--share_level', help='cls & det branch level', default=2, type=int)
    parser.add_argument('--mil_topk', default=1, type=int)
    parser.add_argument('--det_softmax', help='whether or not use softmax over det branch, before, after or no', default='no', type=str)

    parser.add_argument('--prop_method', help='ss, eb, or mcg', default='eb', type=str)
    parser.add_argument('--prop_min_scale', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--num_prop', help='maximum number of proposals to use for training', default=2000, type=int)

    parser.add_argument('--alpha', help='weight for target loss', default=0.5, type=float)
    parser.add_argument('--bs', help='source training batch size', default=128, type=int)
    parser.add_argument('--pos_ratio', help='ratio of positive roi', default=0.25, type=float)

    parser.add_argument('--lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--s', dest='session', help='training session', default=0, type=int)
    parser.add_argument('--seed', help='random sed', default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume', help='resume checkpoint or not', action='store_true')
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=0, type=int)
    parser.add_argument('--checkiter', dest='checkiter', help='checkiter to load model', default=0, type=int)

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

    source_train_dataset = TDETDataset(['coco60_train2014', 'coco60_val2014'], args.data_dir, args.prop_method,
                                       num_classes=60, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)
    target_train_dataset = TDETDataset(['voc07_trainval'], args.data_dir, args.prop_method,
                                       num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

    lr = args.lr

    if args.net == 'TDET_VGG16':
        model = TDET_VGG16(os.path.join(args.data_dir, 'pretrained_model/vgg16_caffe.pth'), 20,
                           pooling_method=args.pooling_method, cls_specific_det=args.cls_specific,
                           backprop2det=args.backprop2det, share_level=args.share_level, mil_topk=args.mil_topk,
                           det_softmax=args.det_softmax, det_choice=args.det_choice)
    else:
        raise Exception('network is not defined')

    optimizer = model.get_optimizer(args.lr)

    if args.resume:
        load_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.checksession, args.checkiter))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        assert args.net == checkpoint['net']
        args.start_iter = checkpoint['iterations'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))

    log_file_name = os.path.join(output_dir, 'log_{}_{}.txt'.format(args.net, args.session))
    if args.resume:
        log_file = open(log_file_name, 'a')
    else:
        log_file = open(log_file_name, 'w')
    log_file.write(str(args))
    log_file.write('\n')

    model.to(device)
    model.train()
    source_loss_sum = 0
    target_loss_sum = 0
    source_pos_prop_sum = 0
    source_neg_prop_sum = 0
    target_prop_sum = 0
    start = time.time()
    for step in range(args.start_iter, args.max_iter + 1):
        if step % len(source_train_dataset) == 1:
            source_rand_perm = np.random.permutation(len(source_train_dataset))
        if step % len(target_train_dataset) == 1:
            target_rand_perm = np.random.permutation(len(target_train_dataset))

        source_index = source_rand_perm[step % len(source_train_dataset)]
        target_index = target_rand_perm[step % len(target_train_dataset)]

        source_batch = source_train_dataset.get_data(source_index, h_flip=np.random.rand() > 0.5, target_im_size=np.random.choice([480, 576, 688, 864, 1200]))
        target_batch = target_train_dataset.get_data(target_index, h_flip=np.random.rand() > 0.5, target_im_size=np.random.choice([480, 576, 688, 864, 1200]))

        source_im_data = source_batch['im_data'].unsqueeze(0).to(device)
        source_proposals = source_batch['proposals']
        source_gt_boxes = source_batch['gt_boxes']
        source_proposals, source_labels, pos_cnt, neg_cnt = sample_proposals(source_gt_boxes, source_proposals, args.bs, args.pos_ratio)
        source_proposals = source_proposals.to(device)
        source_gt_boxes = source_gt_boxes.to(device)
        source_labels = source_labels.to(device)

        target_im_data = target_batch['im_data'].unsqueeze(0).to(device)
        target_proposals = target_batch['proposals'].to(device)
        target_image_level_label = target_batch['image_level_label'].to(device)

        optimizer.zero_grad()

        # source forward & backward
        source_loss = model.forward_det_only(source_im_data, source_proposals, source_labels)
        source_loss_sum += source_loss.item()
        source_loss = source_loss * (1 - args.alpha)
        source_loss.backward()

        # target forward & backward
        _, target_loss = model(target_im_data, target_proposals, target_image_level_label)
        target_loss_sum += target_loss.item()
        target_loss = target_loss * args.alpha
        target_loss.backward()

        clip_gradient(model, 10.0)
        optimizer.step()
        source_pos_prop_sum += pos_cnt
        source_neg_prop_sum += neg_cnt
        target_prop_sum += target_proposals.size(0)

        if step % args.disp_interval == 0:
            end = time.time()
            loss_sum = source_loss_sum * (1 - args.alpha) + target_loss_sum * args.alpha
            loss_sum /= args.disp_interval
            source_loss_sum /= args.disp_interval
            target_loss_sum /= args.disp_interval
            source_pos_prop_sum /= args.disp_interval
            source_neg_prop_sum /= args.disp_interval
            target_prop_sum /= args.disp_interval
            log_message = "[%s][session %d][iter %4d] loss: %.4f, src_loss: %.4f, tar_loss: %.4f, pos_prop: %.1f, neg_prop: %.1f, tar_prop: %.1f, lr: %.2e, time: %.1f" % \
                          (args.net, args.session, step, loss_sum, source_loss_sum, target_loss_sum, source_pos_prop_sum, source_neg_prop_sum, target_prop_sum, lr, end - start)
            print(log_message)
            log_file.write(log_message + '\n')
            log_file.flush()
            source_loss_sum = 0
            target_loss_sum = 0
            source_pos_prop_sum = 0
            source_neg_prop_sum = 0
            target_prop_sum = 0
            start = time.time()

            if args.det_choice > 1:
                choice_weight = repr(F.softmax(model.choice_layer.weight.clone(), 0))
                print(choice_weight)
                log_file.write(choice_weight + '\n')

        if step in (args.max_iter * 4 // 7, args.max_iter * 6 // 7):
            adjust_learning_rate(optimizer, 0.1)
            lr *= 0.1

        if step % args.save_interval == 0 or step == args.max_iter:
            save_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.session, step))
            checkpoint = dict()
            checkpoint['net'] = args.net
            checkpoint['session'] = args.session
            checkpoint['cls_specific'] = args.cls_specific
            checkpoint['pooling_method'] = args.pooling_method
            checkpoint['share_level'] = args.share_level
            checkpoint['det_softmax'] = args.det_softmax
            checkpoint['det_choice'] = args.det_choice
            checkpoint['iterations'] = step
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()

            save_checkpoint(checkpoint, save_name)
            print('save model: {}'.format(save_name))

    log_file.close()


if __name__ == '__main__':
    train()