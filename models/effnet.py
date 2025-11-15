import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Helpers -----

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)  # or use nn.SiLU() if you prefer

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        reduced = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv2d(in_ch, reduced, kernel_size=1)
        self.act = Swish()
        self.fc2 = nn.Conv2d(reduced, in_ch, kernel_size=1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        return x * torch.sigmoid(s)

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck with SE.
    Structure: (optional) 1x1 expand -> depthwise kxk -> SE -> 1x1 project
    Residual if stride=1 and in==out.
    """
    def __init__(self, in_ch, out_ch, stride, expand_ratio, kernel_size=3, se_ratio=0.25, drop_rate=0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = int(in_ch * expand_ratio)

        layers = []
        # Expand
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),
            ]
        else:
            mid_ch = in_ch

        # Depthwise
        layers += [
            nn.Conv2d(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        ]
        self.pre_se = nn.Sequential(*layers)

        # Squeeze-Excite
        self.se = SqueezeExcite(mid_ch, se_ratio=se_ratio)

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(p=drop_rate, inplace=True) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        out = self.pre_se(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            if self.drop_rate > 0.0:
                out = self.dropout(out)
            out = out + x
        return out

# ----- EfficientNet-B0 for CIFAR (32x32 input) -----

class EfficientNetB0_CIFAR(nn.Module):
    """
    EfficientNet-B0 layout with CIFAR-friendly stem (stride=1).
    Stages follow B0 repeats/strides; overall downsamples ~x16 to get 2x2 before GAP.
    """
    def __init__(self, num_classes=100, se_ratio=0.25, drop_rate=0.2):
        super().__init__()
        self.in_channels = 32

        # Stem: 3x3 conv, stride=1 for 32x32 inputs (CIFAR)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.act = nn.SiLU(inplace=True)

        # (expand, out_ch, repeats, stride, k)
        cfgs = [
            # B0 original: (1, 16, 1, 1, 3)
            (1,   16, 1, 1, 3),
            # (6, 24, 2, 2, 3)
            (6,   24, 2, 2, 3),
            # (6, 40, 2, 2, 5)
            (6,   40, 2, 2, 5),
            # (6, 80, 3, 2, 3)
            (6,   80, 3, 2, 3),
            # (6, 112, 3, 1, 5)
            (6,  112, 3, 1, 5),
            # (6, 192, 4, 2, 5)
            (6,  192, 4, 2, 5),
            # (6, 320, 1, 1, 3)
            (6,  320, 1, 1, 3),
        ]

        # Build stages
        stages = []
        for expand, out_ch, repeats, stride, k in cfgs:
            stages.append(self._make_layer(MBConv, out_ch, repeats, stride,
                                           expand_ratio=expand, kernel_size=k, se_ratio=se_ratio))
        self.stages = nn.Sequential(*stages)

        # Head
        self.head_channels = 1280
        self.conv_head = nn.Conv2d(cfgs[-1][1], self.head_channels, kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm2d(self.head_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        self.fc = nn.Linear(self.head_channels, num_classes)

        # Init
        self._init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, expand_ratio, kernel_size, se_ratio):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        in_ch = self.in_channels
        for s in strides:
            layers.append(block(in_ch, out_channels, s, expand_ratio, kernel_size, se_ratio))
            in_ch = out_channels
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.stages(out)

        out = self.conv_head(out)
        out = self.bn_head(out)
        out = self.act(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
