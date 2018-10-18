import os
import numpy as np
import argparse
import time

import torch

from utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient, calc_grad_norm
from utils.box_utils import sample_proposals
from model.cam_det import CamDet
from datasets.tdet_dataset import TDETDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--net', default='CAM_DET', type=str)
    parser.add_argument('--start_iter', help='starting iteration', default=1, type=int)
    parser.add_argument('--max_iter', help='number of iterations', default=70000, type=int)
    parser.add_argument('--disp_interval', help='number of iterations to display loss', default=1000, type=int)
    parser.add_argument('--save_interval', dest='save_interval', help='number of iterations to save', default=10000, type=int)
    parser.add_argument('--save_dir', help='directory to save models', default="../repo/tdet")
    parser.add_argument('--data_dir', help='directory to load data', default='../data', type=str)

    parser.add_argument('--bs', help='training batch size', default=10, type=int)

    parser.add_argument('--lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--s', dest='session', help='training session', default=-1, type=int)
    parser.add_argument('--seed', help='random sed', default=1, type=int)
    parser.add_argument('--target_only', action='store_true')

    parser.add_argument('--hidden_dim', help='hidden layer dim for attention', default=128, type=int)


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
    assert args.bs % 2 == 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_only = args.target_only
    source_train_dataset = TDETDataset(['coco60_train2014', 'coco60_val2014'], args.data_dir, 'eb', num_classes=60)
    target_train_dataset = TDETDataset(['voc07_trainval'], args.data_dir, 'eb', num_classes=20)

    lr = args.lr

    if args.net == 'CAM_DET':
        model = CamDet(os.path.join(args.data_dir, 'pretrained_model/vgg16_caffe.pth') if not args.resume else None, 20 if target_only else 80, args.hidden_dim)
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
        print("loaded checkpoint %s" % (load_name))
        del checkpoint

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
    total_loss_sum = 0
    start = time.time()
    source_rand_perm = None
    target_rand_perm = None
    for step in range(args.start_iter, args.max_iter + 1):
        if source_rand_perm is None or step % len(source_train_dataset) == 1:
            source_rand_perm = np.random.permutation(len(source_train_dataset))
        if target_rand_perm is None or step % len(target_train_dataset) == 1:
            target_rand_perm = np.random.permutation(len(target_train_dataset))

        source_index = source_rand_perm[step % len(source_train_dataset)]
        target_index = target_rand_perm[step % len(target_train_dataset)]

        optimizer.zero_grad()
        if not target_only:
            source_batch = source_train_dataset.get_data(source_index, h_flip=np.random.rand() > 0.5, target_im_size=np.random.choice([480, 576, 688, 864, 1200]))

            source_im_data = source_batch['im_data'].unsqueeze(0).to(device)
            source_gt_labels = source_batch['gt_labels'] + 20
            source_pos_cls = [i for i in range(80) if i in source_gt_labels]
            source_pos_cls = torch.tensor(np.random.choice(source_pos_cls, min(args.bs, len(source_pos_cls)), replace=False), dtype=torch.long, device=device)

            source_loss, _, _ = model(source_im_data, source_pos_cls)
            source_loss_sum += source_loss.item()

        target_batch = target_train_dataset.get_data(target_index, h_flip=np.random.rand() > 0.5,
                                                     target_im_size=np.random.choice([480, 576, 688, 864, 1200]))

        target_im_data = target_batch['im_data'].unsqueeze(0).to(device)
        target_gt_labels = target_batch['gt_labels']
        target_pos_cls = [i for i in range(80) if i in target_gt_labels]
        target_pos_cls = torch.tensor(np.random.choice(target_pos_cls, min(args.bs, len(target_pos_cls)), replace=False), dtype=torch.long, device=device)

        target_loss, _, _, _ = model(target_im_data, target_pos_cls)
        target_loss_sum += target_loss.item()
        if args.target_only:
            total_loss = target_loss
        else:
            total_loss = (source_loss + target_loss) * 0.5
        total_loss.backward()
        total_loss_sum += total_loss.item()
        clip_gradient(model, 10.0)
        optimizer.step()

        if step % args.disp_interval == 0:
            end = time.time()
            total_loss_sum /= args.disp_interval
            source_loss_sum /= args.disp_interval
            target_loss_sum /= args.disp_interval
            log_message = "[%s][session %d][iter %4d] loss: %.8f, src_loss: %.8f, tar_loss: %.8f, lr: %.2e, time: %.1f" % \
                          (args.net, args.session, step, total_loss_sum, source_loss_sum, target_loss_sum, lr, end - start)
            print(log_message)
            log_file.write(log_message + '\n')
            log_file.flush()
            total_loss_sum = 0
            source_loss_sum = 0
            target_loss_sum = 0
            start = time.time()

        if step in (args.max_iter * 4 // 7, args.max_iter * 6 // 7):
            adjust_learning_rate(optimizer, 0.1)
            lr *= 0.1

        if step % args.save_interval == 0 or step == args.max_iter:
            save_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.session, step))
            checkpoint = dict()
            checkpoint['net'] = args.net
            checkpoint['session'] = args.session
            checkpoint['iterations'] = step
            checkpoint['model'] = model.state_dict()

            save_checkpoint(checkpoint, save_name)
            print('save model: {}'.format(save_name))

    log_file.close()


if __name__ == '__main__':
    train()