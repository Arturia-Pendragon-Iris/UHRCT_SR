import torch
import torch.nn as nn
import torch.nn.functional as F


class EncodingBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, batch_norm=True, dropout=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size),
            nn.PReLU(),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_size))
        layers.extend(
            [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(out_size, out_size, kernel_size=kernel_size),
                nn.PReLU(),
            ]
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_size))
        if dropout:
            layers.append(nn.Dropout())
        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):
        return self.encoding_block(input)


class DecodingBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()
        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv = EncodingBlock(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        output1 = F.interpolate(input1, output2.size()[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([output1, output2], 1))


class Getgradientnopadding(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        self.weight_h = nn.Parameter(torch.FloatTensor(kernel_h).view(1, 1, 3, 3), requires_grad=False)
        self.weight_v = nn.Parameter(torch.FloatTensor(kernel_v).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i:i + 1]
            x_i_v = F.conv2d(x_i, self.weight_v.to(x.device), padding=1)
            x_i_h = F.conv2d(x_i, self.weight_h.to(x.device), padding=1)
            x_list.append(torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6))
        return torch.cat(x_list, dim=1)


class Filter_UNet(nn.Module):
    """
    Lightweight convolutional operator OFc that learns to approximate Frangi.

    This keeps the paper's differentiable operator idea, but its supervision is
    generated with the Frangi implementation in conv_filter/filter.py.
    """

    def __init__(self, in_channel=1, num_classes=1, channel=8):
        super().__init__()
        self.get_g_nopadding = Getgradientnopadding()
        self.conv1 = EncodingBlock(in_channel + 1, channel, kernel_size=5)
        self.maxpool1 = nn.Conv2d(channel, channel, kernel_size=2, stride=2)
        self.conv2 = EncodingBlock(channel, 2 * channel, kernel_size=5)
        self.maxpool2 = nn.Conv2d(2 * channel, 2 * channel, kernel_size=2, stride=2)
        self.center = EncodingBlock(2 * channel, 4 * channel)
        self.decode2 = DecodingBlock(4 * channel, 2 * channel, upsampling=True)
        self.decode1 = DecodingBlock(2 * channel, channel, upsampling=True)
        self.decode0 = nn.Conv2d(channel, num_classes, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(num_classes + in_channel, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        grad = self.get_g_nopadding(input)
        conv1 = self.conv1(torch.cat((input, grad), dim=1))
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        center = self.center(maxpool2)
        decode2 = self.decode2(conv2, center)
        decode1 = self.decode1(conv1, decode2)
        decode0 = self.decode0(decode1)
        return torch.sigmoid(self.final(torch.cat((decode0, input), dim=1)))
