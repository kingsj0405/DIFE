"""
Forked from following link:

https://github.com/jamt9000/DVE/blob/master/model/hourglass.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(*args, **kwargs):
    return nn.GroupNorm(16, *args, **kwargs)


class ResidualBottleneckPreactivation(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, make_bn=None):
        super(ResidualBottleneckPreactivation, self).__init__()

        self.bn1 = make_bn(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = make_bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = make_bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class HourglassBlock(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, make_bn):
        super(HourglassBlock, self).__init__()
        self.block = block
        self.layernames = []
        self.num_blocks = num_blocks
        self.planes = planes
        self.outputs = {}
        self.make_bn = make_bn

        self._hour_glass_layers(depth)

    def _make_blocks(self):
        layers = []
        for i in range(0, self.num_blocks):
            layers.append(self.block(self.planes * self.block.expansion, self.planes, make_bn=self.make_bn))
        return nn.Sequential(*layers)

    def _hour_glass_layers(self, n):
        # Recursively build the hourglass layers
        self.layernames.append('layer%d_1' % n)
        setattr(self, self.layernames[-1], self._make_blocks())

        self.layernames.append('mp%d' % n)
        setattr(self, self.layernames[-1], nn.MaxPool2d(2, stride=2))

        self.layernames.append('layer%d_2' % n)
        setattr(self, self.layernames[-1], self._make_blocks())

        if n == 1:
            self.layernames.append('layer%d_4' % n)
            setattr(self, self.layernames[-1], self._make_blocks())
        else:
            self._hour_glass_layers(n - 1)

        self.layernames.append('layer%d_3' % n)
        setattr(self, self.layernames[-1], self._make_blocks())

        self.layernames.append('up%d' % n)
        setattr(self, self.layernames[-1], nn.Upsample(scale_factor=2))

        self.layernames.append('sum%d' % n)
        # setattr(self, self.layernames[-1], lambda x: self.outputs['layer%d_1' % n] + x)
        setattr(self, self.layernames[-1], (torch.add, 'layer%d_1' % n))

    def forward(self, x):
        outputs = {}
        for layer in self.layernames:
            layerfn = getattr(self, layer)
            if isinstance(layerfn, tuple):
                fn, a = layerfn
                x = fn(outputs[a], x)
            else:
                x = layerfn(x)
            if 'layer' in layer and '_1' in layer:
                outputs[layer] = x
        return x


class HourglassNet(nn.Module):
    def __init__(
        self,
        block=ResidualBottleneckPreactivation,
        num_stacks=1,
        num_blocks=4,
        planes_conv1=64,
        planes_block=128,
        planes_hg=128,
        num_output_channels=16,
        use_group_norm=False,
        output_layer_weight_mul=1.,
        scaled_norm_output=False,
        output_as_tensor=False,
        keep_size=False,
    ):
        super(HourglassNet, self).__init__()

        self.block = block
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.planes_conv1 = planes_conv1
        self.planes_block = planes_block  # num planes in the 3x3 conv layers
        self.planes_hg = planes_hg
        self.depth_hg = 4
        self.num_output_channels = num_output_channels
        if use_group_norm:
            self.make_bn = make_gn
        else:
            self.make_bn = nn.BatchNorm2d
        self.output_layer_weight_mul = output_layer_weight_mul
        self.scaled_norm_output = scaled_norm_output
        self.output_as_tensor = output_as_tensor
        self.keep_size = keep_size

        self.conv1 = nn.Conv2d(3, planes_conv1, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = self.make_bn(planes_conv1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_blocks(planes_conv1, planes_conv1, 1)  # 64 -> 64 -> 128
        self.layer2 = self._make_blocks(planes_conv1 * block.expansion, planes_block, 1)  # 128 -> 128 -> 256
        self.layer3 = self._make_blocks(planes_block * block.expansion, planes_hg, 1)  # 256 -> 128 -> 256

        if self.scaled_norm_output:
            self.normscale = nn.Parameter(torch.tensor(20.))

        nch = self.planes_hg * block.expansion
        hg = []
        output_layers = []

        for i in range(num_stacks):
            hg.append(HourglassBlock(block, num_blocks, self.planes_hg, self.depth_hg, make_bn=self.make_bn))
            res = self._make_blocks(nch, self.planes_hg, self.num_blocks)
            bn = self.make_bn(nch)
            conv = nn.Conv2d(nch, nch, kernel_size=1)
            outlayer = nn.Conv2d(nch, self.num_output_channels, kernel_size=1)
            outlayer.weight.data.mul_(self.output_layer_weight_mul)
            if self.keep_size:
                up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                output_layers.append(nn.Sequential(res, conv, bn, self.relu, up, outlayer))
            else:
                output_layers.append(nn.Sequential(res, conv, bn, self.relu, outlayer))

        self.hg = nn.ModuleList(hg)
        self.output_layers = nn.ModuleList(output_layers)

    def _make_blocks(self, inplanes, planes, num_blocks):
        layers = []

        downsample = None
        if inplanes != planes * self.block.expansion:
            downsample = nn.Conv2d(inplanes, planes * self.block.expansion, kernel_size=1, stride=1)

        for i in range(0, num_blocks):
            layers.append(self.block(inplanes, planes, 1, downsample if i == 0 else None, make_bn=self.make_bn))
            if i == 0:
                inplanes = planes * self.block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            x = self.hg[i](x)
            y = self.output_layers[i](x)
            if self.scaled_norm_output:
                y = F.normalize(y, dim=1, p=2) * self.normscale
            out.append(y)

        # XXX just return last output
        out = [out[-1]]
        if self.output_as_tensor:
            out = out[0]

        return out

