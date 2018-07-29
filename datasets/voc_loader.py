from scipy.misc import imread
from scipy.io import loadmat
import numpy as np
import sys
import os
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


class VOCLoader:
    def __init__(self, root, prop_method, year, name):
        self.items = []
        self.name_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        print('VOC %s %s dataset loading...' % (year, name))

        if prop_method == 'ss':
            prop_dir = os.path.join(root, 'voc07_proposals', 'selective_search')
        elif prop_method == 'eb':
            prop_dir = os.path.join(root, 'voc07_proposals', 'edge_boxes_70')
        elif prop_method == 'mcg':
            prop_dir = os.path.join(root, 'voc07_proposals', 'MCG2015')
        else:
            raise Exception('Undefined proposal name')

        rootpath = os.path.join(root, 'VOCdevkit2007', 'VOC' + year)
        for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
            data = {}
            id = line.strip()
            target = ET.parse(os.path.join(rootpath, 'Annotations', id + '.xml'))

            box_set = []
            category_set = []
            for obj in target.iter('object'):
                cls_name = obj.find('name').text.strip().lower()
                bbox = obj.find('bndbox')

                xmin = int(bbox.find('xmin').text) - 1
                ymin = int(bbox.find('ymin').text) - 1
                xmax = int(bbox.find('xmax').text) - 1
                ymax = int(bbox.find('ymax').text) - 1

                category = self.name_to_index[cls_name]
                box_set.append(np.array([xmin, ymin, xmax, ymax], np.float32))
                category_set.append(category)

            data['id'] = id
            data['boxes'] = np.array(box_set)
            data['categories'] = np.array(category_set, np.long)
            data['img_path'] = os.path.join(rootpath, 'JPEGImages', line.strip() + '.jpg')
            data['prop_path'] = os.path.join(prop_dir, 'mat', id[:4], '%s.mat' % id)
            self.items.append(data)

        print('VOC %s %s dataset loading complete' % (year, name))

    def __len__(self):
        return len(self.items)
