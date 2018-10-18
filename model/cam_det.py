import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.box_utils import *
import torchvision
import copy
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling


class CamDet(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=80, label_dim=128):
        super(CamDet, self).__init__()
        self.label_weights = nn.Parameter(torch.randn([num_class, label_dim]) * 0.1)
        self.num_classes = num_class
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create DC VGG16 without pretrained weights")
        else:
            print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
            state_dict = torch.load(pretrained_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.cam_layer = nn.Conv2d(512, num_class, 1)
        self.encode = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, label_dim, 1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # layer 추가할거면, get_optimizer도 수정해야댐
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        for layer in self.encode:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.01)

    def forward(self, im_data, target_cls):
        conv_feat = self.base(im_data)

        # encoded_feat = self.encode(conv_feat)
        # D, H, W = encoded_feat.squeeze().size()
        # M = H * W
        # encoded_feat = encoded_feat.view(D, M)
        # diff = encoded_feat.permute(1, 0).view(1, M, D) - self.label_weights.view(self.num_classes, 1, D)
        # dist = torch.norm(diff, dim=2)
        # score_map = torch.exp(-dist.view(self.num_classes, H, W))
        # cls_score = self.gmp(score_map.unsqueeze(0)).view(self.num_classes)

        score_map = self.cam_layer(conv_feat)
        cls_score = F.sigmoid(self.gap(score_map).view(self.num_classes))

        target = cls_score.new_zeros(self.num_classes)
        for cls in target_cls:
            target[cls] = 1
        loss = F.binary_cross_entropy(cls_score, target)

        return loss, cls_score, score_map.view(self.num_classes, conv_feat.size(2), conv_feat.size(3)), None

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
