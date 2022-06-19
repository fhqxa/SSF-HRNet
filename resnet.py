import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from SSFR import SSFC, SSFT

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=512)
parser.add_argument("-p", "--parent_relation_dim", type=int, default=128)
parser.add_argument("-r", "--fine_relation_dim", type=int, default=128)
parser.add_argument("-w", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=300000)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-l", "--learning_rate", type=float, default=0.1)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-self_method", "--self_method", default='SSFR')

args = parser.parse_args()
FEATURE_DIM = args.feature_dim
FINE_RELATION_DIM = args.fine_relation_dim
PARENT_RELATION_DIM = args.parent_relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, args, block, layers, kernel=3):
        self.args = args
        self.inplanes = 64
        self.kernel = kernel
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.nFeat = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.kernel, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 84*84
        x = self.layer1(x)   # 42*42
        x = self.layer2(x)  #21*21
        x = self.layer3(x)  #11*11
        x = self.layer4(x) #6*6
        return x


class Relation(nn.Module):

    def __init__(self, block, hidden_size, kernel=3):
        self.inplanes = 512 # 512
        self.kernel = kernel
        super(Relation, self).__init__()
        self.conv1 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 512, 1, stride=2)
        self.layer2 = self._make_layer(block, 1024, 1, stride=2)

        self.fc1 = nn.Linear(4096, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.nFeat = 512* block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.kernel, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.layer1(x)  # torch.Size([250, 512, 7, 7])
        x = self.layer2(x)  # torch.Size([250, 1024, 4, 4])
        x = x.view(x.size(0), -1)  # torch.Size([250, 4096])
        x = self.fc1(x)
        # x = F.sigmoid(self.fc2(x))
        x = self.fc2(x)
        # print("fc x:", x) # [250,1]
        return x

class SSFR(nn.Module):


    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args  # 参数信息

        self.encoder = ResNet(args, BasicBlock, [1, 1, 1, 1], kernel=3)  # 实例化resnet网络,特征提取器提取基本表示张量
        self.encoder_dim = 512

        self.SSFR_module = self._make_SSFR_layer(planes=[512, 64, 64, 64, 512])

    # 自相关表示，planes=[512, 64, 64, 64, 512]
    def _make_SSFR_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()

        if self.args.self_method == 'SSFR':

            corr_block = SSFT(kernel_size=kernel_size, padding=padding)
            self_block = SSFC(planes=planes, stride=stride)

        else:
            raise NotImplementedError

        if self.args.self_method == 'SSFR':

             layers.append(corr_block)
        layers.append(self_block)

        return nn.Sequential(*layers)


    def forward(self, input):

        if self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'fc':
            return self.fc_forward(input)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])  # 跨行求平均
        return self.fc(x)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)  # 进行维度扩充


    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if self.args.self_method:
            identity = x
            x = self.SSFR_module(x)
            if self.args.self_method == 'SSFR':
                x = x + identity
            x = F.relu(x, inplace=False)

        if do_gap:
            # 自适应平均池化函数
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x

def resnet12():

    model = SSFR(args).cuda(GPU)
    model.mode = 'encoder'
    print(model)
    return model

def RR(hidden_size):
    model = Relation(BasicBlock, hidden_size=hidden_size, kernel=3)
    return model