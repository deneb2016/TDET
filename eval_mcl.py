import os
import numpy as np
import argparse
import time

import torch

from model.mcl_tdet_vgg16 import MCL_TDET_VGG16
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


def eval():
    print('Called with args:')
    print(args)

    np.random.seed(3)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    eval_kit = voc_eval_kit('test', '2007', os.path.join(args.data_dir, 'VOCdevkit2007'))

    test_dataset = TDETDataset(['voc07_test'], args.data_dir, args.prop_method,
                               num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

    load_name = os.path.join(args.save_dir, 'tdet', '{}.pth'.format(args.model_name))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    if checkpoint['net'] == 'MCL_TDET_VGG16':
        model = MCL_TDET_VGG16(None, 20, pooling_method=checkpoint['pooling_method'],
                               share_level=checkpoint['share_level'], num_group=checkpoint['num_group'])
    else:
        raise Exception('network is not defined')
    model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

    print(F.softmax(model.attention_layer.weight.detach(), 0))

    model.to(device)
    model.eval()

    start = time.time()

    num_images = len(test_dataset)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(20)
    # thresh = 0.1 * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in range(20)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(20)]
    cls_scores_set = [[[] for _ in range(num_images)] for _ in range(20)]
    det_scores_set = [[[] for _ in range(num_images)] for _ in range(20)]

    for index in range(len(test_dataset)):
        scores = 0
        c_scores = 0
        d_scores = 0
        N = 0
        if args.multiscale:
            comb = itertools.product([False, True], [480, 576, 688, 864, 1200])
        else:
            comb = itertools.product([False], [688])
        for h_flip, im_size in comb:
            test_batch = test_dataset.get_data(index, h_flip, im_size)

            im_data = test_batch['im_data'].unsqueeze(0).to(device)
            proposals = test_batch['proposals'].to(device)

            local_scores, local_cls_scores, local_det_scores = model(im_data, proposals)
            local_scores = local_scores.detach().cpu().numpy()
            local_cls_scores = local_cls_scores.detach().cpu().numpy()
            local_det_scores = local_det_scores.detach().cpu().numpy()
            scores = scores + local_scores
            c_scores = c_scores + local_cls_scores
            d_scores = d_scores + local_det_scores
            N += 1
        scores = 100 * scores / N
        c_scores = 10 * c_scores / N
        d_scores = 10 * d_scores / N

        boxes = test_dataset.get_raw_proposal(index)

        for cls in range(20):
            inds = np.where((scores[:, cls] > thresh[cls]))[0]
            cls_scores = scores[inds, cls]
            cls_c_scores = c_scores[inds, cls]
            cls_d_scores = d_scores[inds, cls]
            cls_boxes = boxes[inds].copy()

            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_c_scores = cls_c_scores[top_inds]
            cls_d_scores = cls_d_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]

            # if cls_scores[0] > 10:
            #     print(cls)
            #     plt.imshow(test_batch['raw_img'])
            #     draw_box(cls_boxes[0:10, :])
            #     draw_box(test_batch['gt_boxes'] / test_batch['im_scale'], 'black')
            #     plt.show()

            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[cls], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[cls]) > max_per_set:
                while len(top_scores[cls]) > max_per_set:
                    heapq.heappop(top_scores[cls])
                thresh[cls] = top_scores[cls][0]

            all_boxes[cls][index] = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            cls_scores_set[cls][index] = cls_c_scores
            det_scores_set[cls][index] = cls_d_scores

        if index % 500 == 499:
           print('%d images complete, elapsed time:%.1f' % (index + 1, time.time() - start))

    for j in range(20):
        for i in range(len(test_dataset)):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]
            cls_scores_set[j][i] = cls_scores_set[j][i][inds]
            det_scores_set[j][i] = det_scores_set[j][i][inds]

    if args.multiscale:
        save_name = os.path.join(args.save_dir, 'detection_result', '{}_multiscale.pkl'.format(args.model_name))
    else:
        save_name = os.path.join(args.save_dir, 'detection_result', '{}.pkl'.format(args.model_name))
    pickle.dump({'all_boxes': all_boxes, 'cls': cls_scores_set, 'det': det_scores_set}, open(save_name, 'wb'))

    print('Detection Complete, elapsed time: %.1f', time.time() - start)

    for cls in range(20):
        for index in range(len(test_dataset)):
            dets = all_boxes[cls][index]
            if dets == []:
                continue
            keep = nms(dets, 0.3)
            all_boxes[cls][index] = dets[keep, :].copy()
    print('NMS complete, elapsed time: %.1f', time.time() - start)

    eval_kit.evaluate_detections(all_boxes)


def eval_saved_result():
    eval_kit = voc_eval_kit('test', '2007', os.path.join(args.data_dir, 'VOCdevkit2007'))

    if args.multiscale:
        save_name = os.path.join(args.save_dir, 'detection_result', '{}_multiscale.pkl'.format(args.model_name))
    else:
        save_name = os.path.join(args.save_dir, 'detection_result', '{}.pkl'.format(args.model_name))

    saved_data = pickle.load(open(save_name, 'rb'), encoding='latin1')
    all_boxes = saved_data['all_boxes']

    for cls in range(20):
        for index in range(len(all_boxes[0])):
            dets = all_boxes[cls][index]
            if dets == [] or len(dets) == 0:
                continue
            keep = nms(dets, 0.3)
            all_boxes[cls][index] = dets[keep, :].copy()

    eval_kit.evaluate_detections(all_boxes)


def show():
    test_dataset = TDETDataset(['voc07_test'], args.data_dir, args.prop_method,
                               num_classes=20, prop_min_scale=args.prop_min_scale, prop_topk=args.num_prop)

    if args.multiscale:
        save_name = os.path.join(args.save_dir, 'detection_result', '{}_multiscale.pkl'.format(args.model_name))
    else:
        save_name = os.path.join(args.save_dir, 'detection_result', '{}.pkl'.format(args.model_name))

    saved_data = pickle.load(open(save_name, 'rb'), encoding='latin1')
    all_boxes = saved_data['all_boxes']
    cls_scores = saved_data['cls']
    det_scores = saved_data['det']

    for cls in range(15, 20):
        for index in range(len(all_boxes[0])):
            dets = all_boxes[cls][index]
            if dets == [] or len(dets) == 0:
                continue
            keep = nms(dets, 0.3)
            all_boxes[cls][index] = dets[keep, :].copy()

            c_s = cls_scores[cls][index][keep]
            d_s = det_scores[cls][index][keep]
            print(cls)
            print(all_boxes[cls][index][:10, 4])
            print(c_s[:10])
            print(d_s[:10])

            if all_boxes[cls][index][0, 4] > 10:
                plt.imshow(test_dataset.get_raw_img(index))
                draw_box(all_boxes[cls][index][0:1, :4], 'black')
                draw_box(all_boxes[cls][index][1:2, :4], 'red')
                draw_box(all_boxes[cls][index][2:3, :4], 'green')
                draw_box(all_boxes[cls][index][3:4, :4], 'blue')
                draw_box(all_boxes[cls][index][4:5, :4], 'yellow')
                plt.show()

if __name__ == '__main__':
    eval()
    #eval_saved_result()
    #show()