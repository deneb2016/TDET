import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.box_utils import *
import torchvision
import copy
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling


class AttentionLayer(nn.Module):
    def __init__(self, num_class, input_feat_dim, hidden_feat_dim, sim_metric='cosine'):
        super(AttentionLayer, self).__init__()
        self.label_weights = nn.Parameter(torch.randn([num_class, hidden_feat_dim]) * 0.01)
        self.feat_weights = nn.Parameter(torch.randn([hidden_feat_dim, input_feat_dim]) * 0.01)
        self.sim_metric = sim_metric
        self.D = hidden_feat_dim

    def forward(self, conv_feat, pos_cls, do_softmax=True):
        _, C, H, W = conv_feat.size()
        L = pos_cls.size(0)
        M = H * W
        conv_feat = conv_feat.view(C, M)

        embedded_label = self.label_weights[pos_cls]
        embedded_feat = torch.matmul(self.feat_weights, conv_feat)
        if self.sim_metric == 'dot_product':
            attention = torch.matmul(embedded_label, embedded_feat)
        elif self.sim_metric == 'cosine':
            attention = torch.matmul(embedded_label, embedded_feat)
            label_norm = torch.norm(embedded_label, p=2, dim=1).view(-1, 1)
            feat_norm = torch.norm(embedded_feat, p=2, dim=0).view(1, -1)
            norm = label_norm * feat_norm
            norm = torch.clamp(norm, min=0.0001)
            attention = attention / norm
        elif self.sim_metric == 'l2':
            embedded_label = embedded_label.view(L, self.D, 1).expand(L, self.D, M)
            embedded_feat = embedded_feat.view(1, self.D, M).expand(L, self.D, M)
            attention = -torch.norm(embedded_label - embedded_feat, p=2, dim=1)
        else:
            raise Exception('aaa')

        if do_softmax:
            attention = F.softmax(attention, 1)

        attended_feat = torch.matmul(attention, conv_feat.permute(1, 0))
        return attention.view(L, H, W), attended_feat


class AttentiveDet(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=80, hidden_dim=128, sim_metric='cosine'):
        super(AttentiveDet, self).__init__()
        self.num_classes = num_class
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create DC VGG16 without pretrained weights")
        else:
            print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
            state_dict = torch.load(pretrained_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.attention_layer = AttentionLayer(num_class, 512, hidden_dim, sim_metric)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_class)
        )
        # layer 추가할거면, get_optimizer도 수정해야댐
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        for layer in self.classifier:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.01)

    def forward(self, im_data, target_cls):
        conv_feat = self.base(im_data)
        attention, attended_feat = self.attention_layer(conv_feat, target_cls, do_softmax=True)
        score = self.classifier(attended_feat)
        loss = F.cross_entropy(score, target_cls)

        return F.softmax(score, 1), loss, attention

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


class AttentionLayerGlobal(nn.Module):
    def __init__(self, num_class, input_feat_channel, input_feat_size, hidden_feat_dim):
        super(AttentionLayerGlobal, self).__init__()
        self.label_weights = nn.Parameter(torch.randn([num_class, hidden_feat_dim]) * 0.01)
        self.encode = nn.Linear(input_feat_channel * input_feat_size * input_feat_size, hidden_feat_dim, bias=False)
        self.decode = nn.Linear(hidden_feat_dim, input_feat_size * input_feat_size)

        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.encode, 0, 0.01)
        normal_init(self.decode, 0, 0.01)

    def forward(self, conv_feat, pos_cls, do_softmax=True):
        _, C, H, W = conv_feat.size()
        L = pos_cls.size(0)
        conv_feat = conv_feat.view(1, -1)

        embedded_label = self.label_weights[pos_cls]
        embedded_feat = self.encode(conv_feat)
        attention = self.decode(embedded_label * embedded_feat)
        if do_softmax:
            attention = F.softmax(attention, 1)
        attended_feat = torch.matmul(attention, conv_feat.view(C, H * W).permute(1, 0))
        densified_attention = torch.matmul(attended_feat, conv_feat.view(C, H * W))
        return attention.view(L, H, W), attended_feat, densified_attention.view(L, H, W)


class AttentiveDetGlobal(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=80, hidden_dim=1024, im_size=640):
        super(AttentiveDetGlobal, self).__init__()
        self.num_classes = num_class
        vgg = torchvision.models.vgg16()
        if pretrained_model_path is None:
            print("Create DC VGG16 without pretrained weights")
        else:
            print("Loading pretrained VGG16 weights from %s" % (pretrained_model_path))
            state_dict = torch.load(pretrained_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.attention_layer = AttentionLayerGlobal(num_class, 512, im_size // 16, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_class)
        )
        # layer 추가할거면, get_optimizer도 수정해야댐
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        for layer in self.classifier:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.01)

    def forward(self, im_data, target_cls):
        conv_feat = self.base(im_data)
        attention, attended_feat, densified_attention = self.attention_layer(conv_feat, target_cls, do_softmax=True)
        score = self.classifier(attended_feat)
        loss = F.cross_entropy(score, target_cls)

        return F.softmax(score, 1), loss, attention, densified_attention

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
