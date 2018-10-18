import os
import numpy as np
import argparse
import time

import torch

from model.attentive_det_vgg16 import AttentiveDetGlobal
from datasets.tdet_dataset import TDETDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math
import pickle
from utils.cpu_nms import cpu_nms as nms
import heapq
import itertools
from frcnn_eval.pascal_voc import voc_eval_kit
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--save_dir', help='directory to load model and save detection results', default="../repo/")
    parser.add_argument('--data_dir', help='directory to load data', default='../data', type=str)

    parser.add_argument('--multiscale', action='store_true')

    parser.add_argument('--target_only', action='store_true')
    parser.add_argument('--hidden_dim', help='hidden layer dim for attention', default=1024, type=int)
    parser.add_argument('--im_size', help='im_size', default=640, type=int)

    parser.add_argument('--prop_method', help='ss, eb, or mcg', default='eb', type=str)
    parser.add_argument('--prop_min_scale', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--num_prop', help='maximum number of proposals to use for training', default=2000, type=int)

    parser.add_argument('--model_name', default='ATT_DET_2_70000', type=str)

    args = parser.parse_args()
    return args

args = parse_args()

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


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
def show():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    load_name = os.path.join(args.save_dir, 'tdet', '{}.pth'.format(args.model_name))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    if checkpoint['net'] == 'ATT_DET':
        model = AttentiveDetGlobal(None, 20 if args.target_only else 80, args.hidden_dim, args.im_size)
    else:
        raise Exception('network is not defined')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("loaded checkpoint %s" % (load_name))

    test_dataset = TDETDataset(['voc07_test'], args.data_dir, args.prop_method,
                               num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

    for data_idx in range(len(test_dataset)):
        batch = test_dataset.get_data(data_idx, False, args.im_size, True)
        img = cv2.resize(batch['raw_img'], None, None, fx=batch['im_scale'][0], fy=batch['im_scale'][1], interpolation=cv2.INTER_LINEAR)
        im_data = batch['im_data'].unsqueeze(0).to(device)
        gt_labels = batch['gt_labels']
        pos_cls = [i for i in range(80) if i in gt_labels]
        pos_cls = torch.tensor(pos_cls, dtype=torch.long, device=device)

        if len(pos_cls) < 2:
            continue
        print(pos_cls)
        score, loss, attention, densified_attention = model(im_data, pos_cls)
        for i, cls in enumerate(pos_cls):
            print(VOC_CLASSES[cls])
            print(score[i])
            print(score[i, cls])
            maxi = attention[i].max()
            print(maxi)
            #attention[i][attention[i] < maxi] = 0
            plt.imshow(img)
            plt.show()
            plt.imshow(attention[i].cpu().detach().numpy())

            plt.show()
            plt.imshow(densified_attention[i].cpu().detach().numpy())

            plt.show()

if __name__ == '__main__':
    #eval()
    #eval_saved_result()
    show()