from scipy.misc import imread
import json
import numpy as np
import os
from scipy.io import loadmat


class COCOLoader:
    def __init__(self, root, name, prop_method):
        self.items = []

        anno_path = os.path.join(root, 'coco', 'annotations', '%s.json' % name)
        if 'train' in name:
            img_path = os.path.join(root, 'coco', 'images', 'train2014')
        elif 'val' in name:
            img_path = os.path.join(root, 'coco', 'images', 'val2014')
        else:
            raise Exception('undefined dataset name')

        print('dataset loading...' + anno_path)
        if prop_method == 'ss':
            prop_dir = os.path.join(root, 'coco_proposals', 'selective_search')
        elif prop_method == 'eb':
            prop_dir = os.path.join(root, 'coco_proposals', 'edge_boxes_70')
        elif prop_method == 'mcg':
            prop_dir = os.path.join(root, 'coco_proposals', 'MCG')
        else:
            raise Exception('Undefined proposal name')

        anno = json.load(open(anno_path))
        box_set = {}
        category_set = {}
        cid_to_idx = {}
        #print(anno['categories'])
        for i, cls in enumerate(anno['categories']):
            cid_to_idx[cls['id']] = i

        for i, obj in enumerate(anno['annotations']):
            im_id = obj['image_id']
            if im_id not in box_set:
                box_set[im_id] = []
                category_set[im_id] = []
            category = cid_to_idx[obj['category_id']]

            bbox = np.array(obj['bbox'])
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            box_set[im_id].append(np.array([xmin, ymin, xmax, ymax], np.float32))
            category_set[im_id].append(category)

        for i, img in enumerate(anno['images']):
            data = {}
            id = img['id']
            assert id in box_set and len(box_set[id]) > 0
            assert id in category_set and len(category_set[id]) > 0

            data['id'] = id
            data['boxes'] = np.array(box_set[id])
            data['categories'] = np.array(category_set[id], np.long)
            data['img_path'] = os.path.join(img_path, img['file_name'])
            data['prop_path'] = os.path.join(prop_dir, 'mat', img['file_name'][:14], img['file_name'][:22], '%s.mat' % img['file_name'][:-4])
            self.items.append(data)

        print('dataset loading complete')
        print('%d / %d images' % (len(self.items), len(anno['images'])))

    def __len__(self):
        return len(self.items)

