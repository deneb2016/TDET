import torch.nn as nn
import torch.nn.functional as F
import torch
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from utils.box_utils import *
import torchvision
import copy


class TDET_VGG16(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=20, pooling_method='roi_pooling', cls_specific_det=False, share_level=2, mil_topk=1):
        super(TDET_VGG16, self).__init__()
        assert 0 <= share_level <= 2
        self.num_classes = num_class
        self.cls_specific_det = cls_specific_det
        self.mil_topk = mil_topk
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create WSDDN_VGG16 without pretrained weights")
        else:
            print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
            state_dict = torch.load(pretrained_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        top = list()
        if share_level >= 1:
            top.append(vgg.classifier[0])
            top.append(nn.ReLU(True))

        if share_level == 2:
            top.append(vgg.classifier[3])
            top.append(nn.ReLU(True))

        self.top = nn.Sequential(*top)

        cls = list()
        det = list()

        if share_level == 0:
            cls.append(copy.deepcopy(vgg.classifier[0]))
            cls.append(nn.ReLU(True))
            det.append(copy.deepcopy(vgg.classifier[0]))
            det.append(nn.ReLU(True))

        if share_level <= 1:
            cls.append(copy.deepcopy(vgg.classifier[3]))
            cls.append(nn.ReLU(True))
            det.append(copy.deepcopy(vgg.classifier[3]))
            det.append(nn.ReLU(True))

        cls.append(nn.Linear(4096, self.num_classes))
        if cls_specific_det:
            det.append(nn.Linear(4096, self.num_classes))
        else:
            det.append(nn.Linear(4096, 1))

        self.cls_layer = nn.Sequential(*cls)
        self.det_layer = nn.Sequential(*det)

        if pooling_method == 'roi_pooling':
            self.region_pooling = _RoIPooling(7, 7, 1.0 / 16.0)
        elif pooling_method == 'roi_align':
            self.region_pooling = RoIAlignAvg(7, 7, 1.0 / 16.0)
        else:
            raise Exception('Undefined pooling method')

        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.cls_layer[-1], 0, 0.01, False)
        normal_init(self.det_layer[-1], 0, 0.01, False)

    def forward_shared_feat(self, im_data, rois):
        N = rois.size(0)
        feature_map = self.base(im_data)
        zero_padded_rois = torch.cat([torch.zeros(N, 1).to(rois), rois], 1)
        pooled_feat = self.region_pooling(feature_map, zero_padded_rois).view(N, -1)
        shared_feat = self.top(pooled_feat)
        return shared_feat

    def forward(self, im_data, rois, image_level_label=None):
        shared_feat = self.forward_shared_feat(im_data, rois)
        cls_score = self.cls_layer(shared_feat)
        det_score = self.det_layer(shared_feat)

        cls_score = F.softmax(cls_score, dim=1)
        det_score = F.sigmoid(det_score)

        if self.cls_specific_det:
            scores = cls_score * det_score
        else:
            scores = cls_score * det_score.detach()

        if image_level_label is None:
            return scores

        image_level_scores, _ = torch.topk(scores, self.mil_topk, dim=0)
        image_level_scores = torch.mean(scores, 0)

        loss = F.binary_cross_entropy(image_level_scores, image_level_label.to(torch.float32))

        return scores, loss

    def forward_det_only(self, im_data, rois, gt_labels=None):
        shared_feat = self.forward_shared_feat(im_data, rois)
        det_score = self.det_layer(shared_feat)
        det_score = F.sigmoid(det_score)

        if self.cls_specific_det:
            det_score = torch.mean(det_score, 1)

        det_score = det_score.view(rois.size(0))
        loss = F.binary_cross_entropy(det_score, gt_labels.to(torch.float32))
        return loss
