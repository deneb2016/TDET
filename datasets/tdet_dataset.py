import torch.utils.data as data
import torch
from scipy.io import loadmat
from scipy.misc import imread
import numpy as np
import cv2
from datasets.voc_loader import VOCLoader
from datasets.coco_loader import COCOLoader


class TDETDataset(data.Dataset):
    def __init__(self, dataset_names, data_dir, prop_method, num_classes=20, prop_min_scale=10, prop_topk=2000):
        self._dataset_loaders = []
        self.num_classes = num_classes
        self.prop_min_scale = prop_min_scale
        self.prop_topk = prop_topk
        self.prop_method = prop_method
        for name in dataset_names:
            if name == 'voc07_trainval':
                self._dataset_loaders.append(VOCLoader(data_dir, prop_method, '2007', 'trainval'))
            elif name == 'voc07_test':
                self._dataset_loaders.append(VOCLoader(data_dir, prop_method, '2007', 'test'))
            elif name == 'coco60_train2014':
                self._dataset_loaders.append(COCOLoader(data_dir, 'coco60_train2014', prop_method))
            elif name == 'coco60_val2014':
                self._dataset_loaders.append(COCOLoader(data_dir, 'coco60_val2014', prop_method))
            else:
                raise Exception('Undefined dataset %s' % name)

    def unique_boxes(self, boxes, scale=1.0):
        """Return indices of unique boxes."""
        v = np.array([1, 1e3, 1e6, 1e9])
        hashes = np.round(boxes * scale).dot(v)
        _, index = np.unique(hashes, return_index=True)
        return np.sort(index)

    def select_proposals(self, proposals, scores):
        keep = self.unique_boxes(proposals)
        proposals = proposals[keep]
        scores = scores[keep]
        w = proposals[:, 2] - proposals[:, 0] + 1
        h = proposals[:, 3] - proposals[:, 1] + 1
        keep = np.nonzero((w >= self.prop_min_scale) * (h >= self.prop_min_scale))[0]
        proposals = proposals[keep]
        scores = scores[keep]
        order = np.argsort(-scores)
        order = order[:min(self.prop_topk, order.shape[0])]
        return proposals[order], scores[order]

    def get_data(self, index, h_flip=False, target_im_size=688):
        im, gt_boxes, gt_categories, proposals, prop_scores, id, loader_index = self.get_raw_data(index)
        raw_img = im.copy()
        proposals, prop_scores = self.select_proposals(proposals, prop_scores)
        # rgb -> bgr
        im = im[:, :, ::-1]

        # horizontal flip
        if h_flip:
            im = im[:, ::-1, :]
            raw_img = raw_img[:, ::-1, :].copy()

            flipped_xmin = im.shape[1] - gt_boxes[:, 2]
            flipped_xmax = im.shape[1] - gt_boxes[:, 0]
            gt_boxes[:, 0] = flipped_xmin
            gt_boxes[:, 2] = flipped_xmax

            flipped_xmin = im.shape[1] - proposals[:, 2]
            flipped_xmax = im.shape[1] - proposals[:, 0]
            proposals[:, 0] = flipped_xmin
            proposals[:, 2] = flipped_xmax

        # cast to float type and mean subtraction
        im = im.astype(np.float32, copy=False)
        im -= np.array([[[102.9801, 115.9465, 122.7717]]])

        # image rescale
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = target_im_size / float(im_size_min)

        if im_size_max * im_scale > 2000:
            im_scale = 2000 / im_size_max
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        gt_boxes = gt_boxes * im_scale
        proposals = proposals * im_scale

        # to tensor
        data = torch.tensor(im, dtype=torch.float32)
        data = data.permute(2, 0, 1).contiguous()
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        proposals = torch.tensor(proposals, dtype=torch.float32)
        prop_scores = torch.tensor(prop_scores, dtype=torch.float32)
        gt_categories = torch.tensor(gt_categories, dtype=torch.long)

        image_level_label = torch.zeros(self.num_classes, dtype=torch.uint8)
        for label in gt_categories:
            image_level_label[label] = 1
        return {'im_data': data,
                'gt_boxes': gt_boxes,
                'gt_labels': gt_categories,
                'proposals': proposals,
                'prop_scores': prop_scores,
                'image_level_label': image_level_label,
                'im_scale': im_scale,
                'raw_img': raw_img,
                'id': id}

    def get_raw_proposal(self, index):
        here = None
        loader_index = 0

        # select proper data loader by index
        for loader in self._dataset_loaders:
            if index < len(loader):
                here = loader.items[index]
                break
            else:
                index -= len(loader)
                loader_index += 1

        assert here is not None
        raw_prop = loadmat(here['prop_path'])
        proposals = raw_prop['boxes'].astype(np.float32)
        prop_scores = raw_prop['scores'][:, 0].astype(np.float32)
        if self.prop_method == 'ss':
            prop_scores = -prop_scores

        proposals, prop_scores = self.select_proposals(proposals, prop_scores)
        return proposals

    def get_raw_img(self, index):
        here = None
        loader_index = 0

        # select proper data loader by index
        for loader in self._dataset_loaders:
            if index < len(loader):
                here = loader.items[index]
                break
            else:
                index -= len(loader)
                loader_index += 1

        assert here is not None
        im = imread(here['img_path'])
        return im

    def get_raw_data(self, index):
        here = None
        loader_index = 0

        # select proper data loader by index
        for loader in self._dataset_loaders:
            if index < len(loader):
                here = loader.items[index]
                break
            else:
                index -= len(loader)
                loader_index += 1

        assert here is not None
        im = imread(here['img_path'])

        # gray to rgb
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        gt_boxes = here['boxes'].copy()
        gt_categories = here['categories'].copy()
        raw_prop = loadmat(here['prop_path'])
        proposals = raw_prop['boxes'].astype(np.float32)
        prop_scores = raw_prop['scores'][:, 0].astype(np.float32)
        if self.prop_method == 'ss':
            prop_scores = -prop_scores
        id = here['id']
        return im, gt_boxes, gt_categories, proposals, prop_scores, id, loader_index

    def __len__(self):
        tot_len = 0
        for loader in self._dataset_loaders:
            tot_len += len(loader)
        return tot_len
