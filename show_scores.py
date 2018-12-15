import os
import numpy as np
import argparse
import time

import torch

from model.new_tdet import NEW_TDET
from datasets.tdet_dataset import TDETDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math
import pickle
from utils.cpu_nms import cpu_nms as nms
import heapq
import itertools
from frcnn_eval.pascal_voc import voc_eval_kit

def parse_args():
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--save_dir', help='directory to load model and save detection results', default="../repo/")
    parser.add_argument('--data_dir', help='directory to load data', default='../data', type=str)

    parser.add_argument('--multiscale', action='store_true')

    parser.add_argument('--prop_method', help='ss, eb, or mcg', default='eb', type=str)
    parser.add_argument('--prop_min_scale', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--num_prop', help='maximum number of proposals to use for training', default=2000, type=int)

    parser.add_argument('--model_name', default='', type=str)

    args = parser.parse_args()
    return args

args = parse_args()
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
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


if torch.cuda.is_available():
    torch.cuda.manual_seed(5)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
test_dataset = TDETDataset(['voc07_trainval'], args.data_dir, args.prop_method,
                           num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

load_name = os.path.join(args.save_dir, 'tdet', '{}.pth'.format(args.model_name))
print("loading checkpoint %s" % (load_name))
checkpoint = torch.load(load_name)
if checkpoint['net'] == 'NEW_TDET':
    model = NEW_TDET(None, 20, pooling_method=checkpoint['pooling_method'], share_level=checkpoint['share_level'])
else:
    raise Exception('network is not defined')
model.load_state_dict(checkpoint['model'])
print("loaded checkpoint %s" % (load_name))

model.to(device)
model.eval()

cls = 10
for index in range(len(test_dataset)):
    batch = test_dataset.get_data(index, False, 640)
    if batch['image_level_label'][cls] == 0:
        continue
    im_data = batch['im_data'].unsqueeze(0).to(device)

    # proposals = [
    #     [100, 100, 200, 200]
    # ]
    # prop_tensor = torch.tensor(proposals, dtype=torch.float, device=device)
    # prop_tensor = prop_tensor * batch['im_scale']

    prop_tensor = batch['proposals'].to(device)
    _, c_scores, d_scores = model(im_data, prop_tensor)
    c_scores = c_scores.detach().cpu().numpy()
    d_scores = d_scores.detach().cpu().numpy()

    c_sorted_indices = np.argsort(-c_scores[:, cls])
    d_sorted_indices = np.argsort(-d_scores[:, 0])

    plt.imshow(
        batch['raw_img'])
    draw_box(prop_tensor[c_sorted_indices[:3]] / batch['im_scale'], 'red')
    draw_box(prop_tensor[c_sorted_indices[3:6]] / batch['im_scale'], 'green')
    draw_box(prop_tensor[c_sorted_indices[6:9]] / batch['im_scale'], 'blue')
    plt.show()

    prop_tensor = batch['gt_boxes'].to(device)
    local_scores, local_cls_scores, local_det_scores = model(im_data, prop_tensor)

    for i in range(len(prop_tensor)):
        plt.imshow(batch['raw_img'])
        draw_box(prop_tensor[i:i+1, :] / batch['im_scale'], 'r')
        print(local_cls_scores[i, cls])
        print(local_det_scores[i])
        plt.show()

