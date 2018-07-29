from datasets.tdet_dataset import TDETDataset
from matplotlib import pyplot as plt
import numpy as np
from utils.box_utils import all_pair_iou


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


dataset = TDETDataset(dataset_names=['coco60_val'], data_dir='../data', prop_method='mcg', prop_min_size=0, prop_topk=3000, num_classes=60)
tot = 0.0
det = 0.0
for i in range(len(dataset)):
    here = dataset.get_data(i)

    iou = all_pair_iou(here['gt_boxes'], here['proposals'])
    det += iou.max(1)[0].gt(0.8).sum().item()
    tot += iou.size(0)
    recall = det / tot
    if i % 100 == 99:
        print('%d: %f, %f, %.3f' % (i + 1, det, tot, recall))
