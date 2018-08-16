import os
import numpy as np
import argparse
import time

import torch

from model.dc_vgg16 import DC_VGG16
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
    parser.add_argument('--specific_from', help='cls specific layer', default=3, type=int)
    parser.add_argument('--specific_to', help='cls specific layer', default=4, type=int)

    parser.add_argument('--prop_method', help='ss, eb, or mcg', default='eb', type=str)
    parser.add_argument('--prop_min_scale', help='minimum proposal box size', default=20, type=int)
    parser.add_argument('--num_prop', help='maximum number of proposals to use for training', default=2000, type=int)

    parser.add_argument('--model_name', default='DC_VGG16_0_70000', type=str)

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


def show():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    load_name = os.path.join(args.save_dir, 'tdet', '{}.pth'.format(args.model_name))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    if checkpoint['net'] == 'DC_VGG16':
        model = DC_VGG16(os.path.join(args.data_dir, 'pretrained_model/vgg16_caffe.pth'), 80, args.specific_from, args.specific_to)
    else:
        raise Exception('network is not defined')
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()
    print("loaded checkpoint %s" % (load_name))
    print(torch.cuda.device_count())

    test_dataset = TDETDataset(['voc07_test'], args.data_dir, args.prop_method,
                               num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

    for data_idx in range(len(test_dataset)):
        batch = test_dataset.get_data(data_idx, False, 600)
        im_data = batch['im_data'].unsqueeze(0).to(device)
        gt_labels = batch['gt_labels']
        pos_cls = [i for i in range(80) if i in gt_labels]
        print(pos_cls)

        print(model.inference(im_data, [i for i in range(6)]))
        # for cls in range(20):
        #     print('%.2f' % model.inference(im_data, [cls])[0].item())
        plt.imshow(batch['raw_img'])

        plt.show()

if __name__ == '__main__':
    #eval()
    #eval_saved_result()
    show()