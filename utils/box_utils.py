import torch
import numpy as np

def element_wise_iou(boxes_a, boxes_b):
    """
    Compute the element wise IoU
    :param box_a: (n, 4) minmax form boxes
    :param box_b: (n, 4) minmax form boxes
    :return: (n) iou
    """
    max_xy = torch.min(boxes_a[:, 2:], boxes_b[:, 2:])
    min_xy = torch.max(boxes_a[:, :2], boxes_b[:, :2])
    inter_wh = torch.clamp((max_xy - min_xy + 1), min=0)
    I = inter_wh[:, 0] * inter_wh[:, 1]
    A = (boxes_a[:, 2] - boxes_a[:, 0] + 1) * (boxes_a[:, 3] - boxes_a[:, 1] + 1)
    B = (boxes_b[:, 2] - boxes_b[:, 0] + 1) * (boxes_b[:, 3] - boxes_b[:, 1] + 1)
    U = A + B - I
    return I / U


def all_pair_iou(boxes_a, boxes_b):
    """
    Compute the IoU of all pairs.
    :param boxes_a: (n, 4) minmax form boxes
    :param boxes_b: (m, 4) minmax form boxes
    :return: (n, m) iou of all pairs of two set
    """

    N = boxes_a.size(0)
    M = boxes_b.size(0)
    max_xy = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(N, M, 2), boxes_b[:, 2:].unsqueeze(0).expand(N, M, 2))
    min_xy = torch.max(boxes_a[:, :2].unsqueeze(1).expand(N, M, 2), boxes_b[:, :2].unsqueeze(0).expand(N, M, 2))
    inter_wh = torch.clamp((max_xy - min_xy + 1), min=0)
    I = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    A = ((boxes_a[:, 2] - boxes_a[:, 0] + 1) * (boxes_a[:, 3] - boxes_a[:, 1] + 1)).unsqueeze(1).expand_as(I)
    B = ((boxes_b[:, 2] - boxes_b[:, 0] + 1) * (boxes_b[:, 3] - boxes_b[:, 1] + 1)).unsqueeze(0).expand_as(I)
    U = A + B - I

    return I / U


def transform(boxes, transform_param):
    """
    transform boxes
    :param boxes: (n, 4) tensor, (cx, cy, w, h) form.
    :param transform_param: (n, 4) tensor.
    :return: (n, 4) transformed boxes, (cx, cy, w, h) form.
    """

    cx = boxes[:, 0] + transform_param[:, 0] * boxes[:, 2]
    cy = boxes[:, 1] + transform_param[:, 1] * boxes[:, 3]
    w = boxes[:, 2] * torch.exp(transform_param[:, 2])
    h = boxes[:, 3] * torch.exp(transform_param[:, 3])

    return torch.stack([cx, cy, w, h])


def to_cwh_form(boxes):
    """
    :param boxes: (n, 4) tensor, (cx, cy, w, h) form.
    :return: (n, 4) tensor, (xmin, ymin, xmax, ymax) form
    """

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    return torch.stack([cx, cy, w, h])


def to_minmax_form(boxes):
    """
    :param boxes: (n, 4) tensor, (xmin, ymin, xmax, ymax) form.
    :return: (n, 4) tensor, (cx, cy, w, h) form
    """

    xmin = boxes[:, 0] - boxes[:, 2] / 2 + 0.5
    ymin = boxes[:, 1] - boxes[:, 3] / 2 + 0.5
    xmax = boxes[:, 0] + boxes[:, 2] / 2 - 0.5
    ymax = boxes[:, 1] + boxes[:, 3] / 2 - 0.5
    return torch.stack([xmin, ymin, xmax, ymax])


def sample_proposals(gt_boxes, proposals, max_cnt, pos_ratio):
    iou = all_pair_iou(proposals, gt_boxes)
    iou, _ = iou.max(1)

    pos_indices = torch.nonzero(iou.gt(0.5)).squeeze()
    neg_indices = torch.nonzero(iou.gt(0.1) * iou.lt(0.5)).squeeze()

    if pos_indices.dim() == 0:
        pos_cnt = 0
    else:
        pos_cnt = pos_indices.size(0)

    if neg_indices.dim() == 0:
        neg_cnt = 0
    else:
        neg_cnt = neg_indices.size(0)

    pos_cnt = min(pos_cnt, int(max_cnt * pos_ratio))
    neg_cnt = min(neg_cnt, max_cnt - pos_cnt)

    selected_proposals = []
    target_labels = []

    if pos_cnt > 0:
        pos_indices = torch.LongTensor(np.random.choice(pos_indices.numpy(), pos_cnt, replace=False))
        selected_proposals.append(proposals[pos_indices])
        target_labels.append(torch.ones(pos_cnt))

    if neg_cnt > 0:
        neg_indices = torch.LongTensor(np.random.choice(neg_indices.numpy(), neg_cnt, replace=False))
        selected_proposals.append(proposals[neg_indices])
        target_labels.append(torch.zeros(neg_cnt))

    if len(selected_proposals) == 0:
        selected_proposals = proposals[:1, :]
        target_labels = torch.zeros(1)
        pos_cnt = 0
        neg_cnt = 1
    else:
        selected_proposals = torch.cat(selected_proposals)
        target_labels = torch.cat(target_labels)

    return selected_proposals, target_labels, pos_cnt, neg_cnt
