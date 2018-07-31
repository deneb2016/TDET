import torch.nn as nn
import torch.nn.functional as F
import torch


class WeightDetLayer(nn.Module):
    def __init__(self, K, C):
        super(WeightDetLayer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(K, C).nomal(0, 0.01))

    def forward(self, det_score):
        w = F.softmax(self.weight, 0)
        weighted_det = torch.matmul(det_score, w)
        return weighted_det
