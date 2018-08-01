import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionLayer(nn.Module):
    def __init__(self, K, C, initial_val=0):
        super(AttentionLayer, self).__init__()
        self.weight = nn.Parameter(torch.full([K, C], initial_val))
        #self.weight = nn.Parameter(torch.zeros(K, C).normal_(0, 0.01))

    def forward(self, det_score):
        w = F.softmax(self.weight, 0)
        weighted_det = torch.matmul(det_score, w)
        return weighted_det
