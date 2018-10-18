import torch

def dist():
    a = torch.rand(3, 2)
    b = torch.rand(4, 3)
    D = a.size(0)
    M = a.size(1)
    K = b.size(0)

    print(a, a.size())
    print(b, b.size())
    diff = a.permute(1, 0).view(1, M, D) - b.view(K, 1, D)
    dist = torch.norm(diff, dim=2)
    print(dist)
    print(dist.size())


dist()