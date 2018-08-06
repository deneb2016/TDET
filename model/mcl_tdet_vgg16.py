import torch.nn as nn
import torch.nn.functional as F
import torch
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from utils.box_utils import *
import torchvision
import copy
from model.attention_layer import AttentionLayer


class MCL_TDET_VGG16(nn.Module):
    def __init__(self, pretrained_model_path=None, num_class=20, pooling_method='roi_pooling', share_level=2, mil_topk=1, num_group=1, attention_lr=1.0, mcl_topk=1):
        super(MCL_TDET_VGG16, self).__init__()
        assert 0 <= share_level <= 2
        self.num_classes = num_class
        self.mil_topk = mil_topk
        self.num_group = num_group
        self.attention_layer = AttentionLayer(num_group, num_class)
        self.attention_lr = attention_lr
        self.mcl_topk = mcl_topk

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
        det.append(nn.Linear(4096, num_group))

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
                if m.bias is not None:
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

        # cut backward connection
        det_score = det_score.detach()

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

    def forward_det_img_level(self, im_data, rois, objectness_labels):
        N = rois.size(0)
        shared_feat = self.forward_shared_feat(im_data, rois)
        det_score = self.det_layer(shared_feat)
        objectness_labels = objectness_labels.view(N, 1).expand(det_score.size())
        all_loss = F.binary_cross_entropy_with_logits(det_score, objectness_labels.to(torch.float32), reduce=False)

        per_group_loss = torch.mean(all_loss, 0)
        loss, selected_group = torch.min(per_group_loss, 0)
        selected_group = selected_group.item()
        return loss, selected_group

    def forward_det_img_level_topk(self, im_data, rois, objectness_labels):
        N = rois.size(0)
        shared_feat = self.forward_shared_feat(im_data, rois)
        det_score = self.det_layer(shared_feat)
        objectness_labels = objectness_labels.view(N, 1).expand(det_score.size())
        all_loss = F.binary_cross_entropy_with_logits(det_score, objectness_labels.to(torch.float32), reduce=False)

        per_group_loss = torch.mean(all_loss, 0)
        sorted_loss, sorted_group = torch.sort(per_group_loss, 0)
        loss = sorted_loss[:-1].mean()
        selected_group = sorted_group[-1].item()
        return loss, selected_group

    def forward_det_obj_level_topk(self, im_data, rois, objectness_labels, box_labels):
        N = rois.size(0)
        shared_feat = self.forward_shared_feat(im_data, rois)
        det_score = self.det_layer(shared_feat)
        objectness_labels = objectness_labels.view(N, 1).expand(det_score.size())
        all_loss = F.binary_cross_entropy_with_logits(det_score, objectness_labels.to(torch.float32), reduce=False)
        loss = 0
        selected_group_cnt = np.zeros(self.num_group)
        for box_id in range(torch.max(box_labels).item() + 1):
            mask = box_labels.eq(box_id)
            if mask.sum() == 0:
                continue
            prop_indices = mask.nonzero().view(-1)
            this_box_loss = all_loss[prop_indices]
            this_box_per_group_loss = torch.sum(this_box_loss, 0)
            sorted_loss, sorted_group = torch.sort(this_box_per_group_loss, 0)
            local_loss = sorted_loss[:self.mcl_topk].sum()
            loss = loss + local_loss
            for i in range(self.mcl_topk):
                selected_group_cnt[sorted_group[i]] += 1
        loss = loss / (N * self.mcl_topk)
        return loss, selected_group_cnt

    def forward_det_obj_level_share_negative(self, im_data, rois, objectness_labels, k):
        N = rois.size(0)
        shared_feat = self.forward_shared_feat(im_data, rois)
        det_score = self.det_layer(shared_feat)
        objectness_labels = objectness_labels.view(N, 1).expand(det_score.size())
        all_loss = F.binary_cross_entropy_with_logits(det_score, objectness_labels.to(torch.float32), reduce=False)

        neg_loss = all_loss[objectness_labels.eq(0)]
        selected_group_cnt = np.zeros(self.num_group)

        if objectness_labels.sum() > 0:
            pos_loss = all_loss[objectness_labels.eq(1)].view(-1, self.num_group)
            sorted_pos_loss, sorted_index = torch.sort(pos_loss, 1)
            selected_pos_loss = sorted_pos_loss[:, 0:k]
            selected_index = sorted_index[:, 0:k]
            for i in range(self.num_group):
                selected_group_cnt[i] = torch.sum(selected_index == i).item()
            loss = (selected_pos_loss.sum() + neg_loss.sum()) / (selected_pos_loss.numel() + neg_loss.numel())
        else:
            loss = neg_loss.mean()
        return loss, selected_group_cnt

    def get_optimizer(self, init_lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                lr = init_lr
                weight_decay = 0.0005
                if 'det_layer' in key:
                    lr = lr * self.mcl_topk
                if 'bias' in key:
                    lr = lr * 2
                    weight_decay = 0
                if 'attention_layer' in key:
                    lr = self.attention_lr
                    weight_decay = 0
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        optimizer = torch.optim.SGD(params, momentum=0.9)
        return optimizer