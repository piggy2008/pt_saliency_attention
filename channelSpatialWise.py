import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelWiseBlock(nn.Module):
    def __init__(self, in_channel, reduction=64):
        super(ChannelWiseBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y

class SpatialWiseBlock(nn.Module):
    def __init__(self, in_channel):
        super(SpatialWiseBlock, self).__init__()
        c = in_channel // 16
        self.conv_in = nn.Conv2d(in_channel, c, kernel_size=1)
        self.conv_out = nn.Conv2d(c, 1, kernel_size=1)

        #encoder
        self.conv1 = nn.Conv2d(c, 2 * c, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(2 * c)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(2 * c, 4 * c, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(4 * c)

        #dencoder
        self.deconv1 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(2 * c)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(2* c, c, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(c)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.conv_in(x)
        y = self.conv1(y)
        y = F.relu(self.bn1(y), inplace=True)
        size = y.size()
        y, indices = self.pool1(y)
        y = self.conv2(y)
        y = F.relu(self.bn2(y), inplace=True)

        y = self.deconv1(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.unpool1(y, indices, size)
        y = self.deconv2(y)
        y = F.relu(self.bn4(y), inplace=True)

        y = self.conv_out(y)
        return x * y