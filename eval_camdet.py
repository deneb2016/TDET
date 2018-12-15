import os
import numpy as np
import argparse
import time

import torch

from model.cam_det import CamDet
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
    parser.add_argument('--hidden_dim', help='hidden layer dim for attention', default=128, type=int)

    parser.add_argument('--prop_method', help='ss, eb, or mcg', default='eb', type=str)
    parser.add_argument('--prop_min_scale', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--num_prop', help='maximum number of proposals to use for training', default=2000, type=int)

    parser.add_argument('--model_name', default='CAM_DET_2_70000', type=str)

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
    if checkpoint['net'] == 'CAM_DET':
        model = CamDet(None, 20 if args.target_only else 80, args.hidden_dim)
    else:
        raise Exception('network is not defined')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("loaded checkpoint %s" % (load_name))

    print(model.label_weights)
    test_dataset = TDETDataset(['voc07_trainval'], args.data_dir, args.prop_method,
                               num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

    for data_idx in range(len(test_dataset)):
        batch = test_dataset.get_data(data_idx, False, 640)
        img = cv2.resize(batch['raw_img'], None, None, fx=batch['im_scale'], fy=batch['im_scale'], interpolation=cv2.INTER_LINEAR)
        im_data = batch['im_data'].unsqueeze(0).to(device)
        gt_labels = batch['gt_labels']
        pos_cls = [i for i in range(80) if i in gt_labels]
        pos_cls = torch.tensor(pos_cls, dtype=torch.long, device=device)

        if 10 not in pos_cls:
            continue

        print(pos_cls)
        loss, score, attention, dist = model(im_data, pos_cls)
        for i, cls in enumerate(pos_cls):
            if cls != 10:
                continue
            print(score)
            print(VOC_CLASSES[cls])
            print(score[cls])

            attention[cls] = attention[cls] /
            maxi = attention[cls].max()
            print(maxi)

            plt.imshow(img)
            plt.show()

            plt.imshow(attention[cls].cpu().detach().numpy())

            plt.show()

if __name__ == '__main__':
    #eval()
    #eval_saved_result()
    show()