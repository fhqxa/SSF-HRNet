# code is based on http://dahyun-kang.github.io

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

class SSFC(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(SSFC, self).__init__()
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[2]),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[3]),
                                   nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))

    def forward(self, x):

        b, c, h, w, u, v = x.shape

        x = x.view(b, c, h * w, u * v)
        x = self.conv1x1_in(x)
        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)
        x = self.conv2(x)
        c = x.shape[1]  # 64

        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)
        return x

class SSFT(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SSFT, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x
        x = self.unfold(x)
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity.unsqueeze(2).unsqueeze(2)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()
        return x


