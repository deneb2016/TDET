import torch.nn as nn
import torch.nn.functional as F
import torch
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from utils.box_utils import *
import torchvision
import copy
from model.attention_layer import AttentionLayer


class SM_MCL_TDET_VGG16(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=20, pooling_method='roi_pooling', share_level=2, mil_topk=1, num_group=1, use_predefined_group=False, attention_lr=1.0):
        super(SM_MCL_TDET_VGG16, self).__init__()
        assert 0 <= share_level <= 2
        self.num_classes = num_class
        self.mil_topk = mil_topk
        self.num_group = num_group
        self.use_predefined_group = use_predefined_group
        self.attention_layer = AttentionLayer(num_group, num_class)
        self.attention_lr = attention_lr

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

        # for background group
        det.append(nn.Linear(4096, num_group + 1))

        self.cls_layer = nn.Sequential(*cls)
        self.det_layer = nn.Sequential(*det)

        if pooling_method == 'roi_pooling':
            self.region_pooling = _RoIPooling(7, 7, 1.0 / 16.0)
        elif pooling_method == 'roi_align':
            self.region_pooling = RoIAlignAvg(7, 7, 1.0 / 16.0)
        else:
            raise Exception('Undefined pooling method')

        # layer 추가할거면, get_optimizer도 수정해야댐
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
        det_score = F.softmax(det_score, dim=1)

        # remove background score and cut backward connection
        det_score = det_score[:, 1:].detach()

        # apply attention
        det_score = self.attention_layer(det_score)
        scores = cls_score * det_score

        if image_level_label is None:
            return scores, cls_score, det_score

        image_level_scores, _ = torch.topk(scores, min(self.mil_topk, rois.size(0)), dim=0)
        image_level_scores = torch.mean(image_level_scores, 0)

        # to avoid numerical error
        image_level_scores = torch.clamp(image_level_scores, min=0, max=1)
        loss = F.binary_cross_entropy(image_level_scores, image_level_label.to(torch.float32))

        return scores, loss

    def forward_det_only(self, im_data, rois, gt_labels=None, predefined_groups=None):
        N = rois.size(0)
        shared_feat = self.forward_shared_feat(im_data, rois)
        det_score = self.det_layer(shared_feat)

        if self.use_predefined_group:
            groups = predefined_groups + 1
            neg_mask = gt_labels.eq(0)
            groups[neg_mask] = 0
        else:
            groups = det_score.new_zeros(size=[N], dtype=torch.long)
            pos_mask = gt_labels.eq(1)
            _, max_score_indices = torch.max(det_score[:, 1:], 1)
            max_score_indices = max_score_indices + 1
            groups[pos_mask] = max_score_indices[pos_mask]

        loss = F.cross_entropy(det_score, groups)
        return loss

    def get_optimizer(self, init_lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                lr = init_lr
                weight_decay = 0.0005
                if 'bias' in key:
                    lr = lr * 2
                    weight_decay = 0
                if 'attention_layer' in key:
                    lr = self.attention_lr
                    weight_decay = 0
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        optimizer = torch.optim.SGD(params, momentum=0.9)
        return optimizer