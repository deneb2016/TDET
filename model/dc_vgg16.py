import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.box_utils import *
import torchvision
import copy
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling


class DC_VGG16_CLS(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=80, bs=16, specific_from=3, specific_to=3):
        super(DC_VGG16_CLS, self).__init__()
        assert specific_from <= specific_to
        self.num_classes = num_class
        self.specific_from = specific_from
        self.specific_to = specific_to
        self.bs = bs
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create DC VGG16 without pretrained weights")
        else:
            print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
            state_dict = torch.load(pretrained_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        up2conv_x = [0, 5, 10, 17, 24, 31]
        self.C1 = nn.Sequential(*list(vgg.features._modules.values())[:up2conv_x[specific_from - 1]])
        self.C2 = nn.ModuleList()
        for i in range(num_class):
            self.C2.append(copy.deepcopy(nn.Sequential(*list(vgg.features._modules.values())[up2conv_x[specific_from - 1]:up2conv_x[specific_to]])))
        self.C3 = nn.Sequential(*list(vgg.features._modules.values())[up2conv_x[specific_to]:])

        self.extra_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, 1)
        # layer 추가할거면, get_optimizer도 수정해야댐
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.extra_conv[0], 0, 0.01, False)
        normal_init(self.classifier, 0, 0.01, False)

    def forward(self, im_data, target_cls, target_labels=None):
        feat = self.C1(im_data)
        spe_feat = self.forward_specific_feat(feat, target_cls)
        cls_feat = self.C3(spe_feat)
        cls_feat = self.extra_conv(cls_feat)
        cls_feat = self.gap(cls_feat).view(-1, 1024)
        scores = self.classifier(cls_feat).view(-1)
        if target_labels is None:
            return F.sigmoid(scores), spe_feat
        else:
            loss = F.binary_cross_entropy_with_logits(scores, target_labels)
            return loss

    def forward_specific_feat(self, feat, target_cls):
        cls_feat = []
        for cls in target_cls:
            cls_feat.append(self.C2[cls](feat))

        cls_feat = torch.cat(cls_feat)
        return cls_feat

    def get_optimizer(self, init_lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                lr = init_lr
                weight_decay = 0.0005
                if 'C2' in key:
                    lr = lr * (self.bs / 4)
                if 'bias' in key:
                    lr = lr * 2
                    weight_decay = 0
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        optimizer = torch.optim.SGD(params, momentum=0.9)
        return optimizer


class DC_VGG16_DET(nn.Module):
    def __init__(self, pretrained_base_model, pooling_method='roi_pooling'):
        super(DC_VGG16_DET, self).__init__()
        self.base = pretrained_base_model

        self.top = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )
        self.det_layer = nn.Linear(4096, 1)
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

        normal_init(self.det_layer, 0, 0.01, False)
        for layer in self.top:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.01, False)

    def get_optimizer(self, init_lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                lr = init_lr
                weight_decay = 0.0005
                if 'bias' in key:
                    lr = lr * 2
                    weight_decay = 0
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        optimizer = torch.optim.SGD(params, momentum=0.9)
        return optimizer

    def forward(self, im_data, target_cls, rois, gt_labels=None):
        N = rois.size(0)
        feat = self.base.C1(im_data).detach()
        specific_feat = self.base.forward_specific_feat(feat, [target_cls]).detach()
        base_feat = self.base.C3(specific_feat)
        zero_padded_rois = torch.cat([torch.zeros(N, 1).to(rois), rois], 1)
        region_feat = self.region_pooling(base_feat, zero_padded_rois).view(N, -1)
        region_feat = self.top(region_feat)
        det_score = self.det_layer(region_feat)

        if gt_labels is None:
            return F.sigmoid(det_score).view(-1)
        else:
            return F.binary_cross_entropy_with_logits(det_score.view(N), gt_labels.to(torch.float))