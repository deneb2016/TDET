import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.box_utils import *
import torchvision
import copy


class DC_VGG16_FEAT(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=80, specific_from=3, specific_to=3):
        super(DC_VGG16_FEAT, self).__init__()
        assert specific_from <= specific_to
        self.num_classes = num_class
        self.specific_from = specific_from
        self.specific_to = specific_to
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create WSDDN_VGG16 without pretrained weights")
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

        normal_init(self.extra_conv[0], 0, 0.001, False)
        normal_init(self.classifier, 0, 0.001, False)

    def forward(self, im_data, pos_cls, neg_cls):
        target_labels = torch.cat([torch.ones(len(pos_cls)), torch.zeros(len(neg_cls))]).cuda()

        feat = self.C1(im_data)
        cls_feat = []
        for cls in pos_cls:
            cls_feat.append(self.C2[cls](feat))
        for cls in neg_cls:
            cls_feat.append(self.C2[cls](feat))

        cls_feat = torch.cat(cls_feat)
        cls_feat = self.C3(cls_feat)
        cls_feat = self.extra_conv(cls_feat)
        cls_feat = self.gap(cls_feat).view(-1, 1024)
        scores = self.classifier(cls_feat).view(-1)

        loss = F.binary_cross_entropy_with_logits(scores, target_labels)
        return loss

    def inference(self, im_data, target_cls):
        feat = self.C1(im_data)
        cls_feat = []
        for cls in target_cls:
            cls_feat.append(self.C2[cls](feat))

        cls_feat = torch.cat(cls_feat)
        cls_feat = self.C3(cls_feat)
        cls_feat = self.extra_conv(cls_feat)
        cls_feat = self.gap(cls_feat).view(-1, 1024)
        scores = self.classifier(cls_feat).view(-1)
        scores = F.sigmoid(scores)

        return scores

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


class DC_VGG16_DET(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=80, specific_from=3, specific_to=3):
        super(DC_VGG16_DET, self).__init__()
        assert specific_from <= specific_to
        self.num_classes = num_class
        self.specific_from = specific_from
        self.specific_to = specific_to
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create WSDDN_VGG16 without pretrained weights")
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

        normal_init(self.extra_conv[0], 0, 0.001, False)
        normal_init(self.classifier, 0, 0.001, False)

    def forward(self, im_data, pos_cls, neg_cls):
        target_labels = torch.cat([torch.ones(len(pos_cls)), torch.zeros(len(neg_cls))]).cuda()

        feat = self.C1(im_data)
        cls_feat = []
        for cls in pos_cls:
            cls_feat.append(self.C2[cls](feat))
        for cls in neg_cls:
            cls_feat.append(self.C2[cls](feat))

        cls_feat = torch.cat(cls_feat)
        cls_feat = self.C3(cls_feat)
        cls_feat = self.extra_conv(cls_feat)
        cls_feat = self.gap(cls_feat).view(-1, 1024)
        scores = self.classifier(cls_feat).view(-1)

        loss = F.binary_cross_entropy_with_logits(scores, target_labels)
        return loss

    def inference(self, im_data, target_cls):
        feat = self.C1(im_data)
        cls_feat = []
        for cls in target_cls:
            cls_feat.append(self.C2[cls](feat))

        cls_feat = torch.cat(cls_feat)
        cls_feat = self.C3(cls_feat)
        cls_feat = self.extra_conv(cls_feat)
        cls_feat = self.gap(cls_feat).view(-1, 1024)
        scores = self.classifier(cls_feat).view(-1)
        scores = F.sigmoid(scores)

        return scores

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