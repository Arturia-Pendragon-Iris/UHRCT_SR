import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class encoding_block(nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """

    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout=False,
    ):
        super().__init__()

        if batch_norm:

            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding,
                          stride=stride, dilation=dilation),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
            ]

        else:
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
            ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=False):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=(1, 1)),
            )

        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=(2, 2), stride=(2, 2))

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.upsample(input1, output2.size()[2:], mode="bilinear")

        return self.conv(torch.cat([output1, output2], 1))


class Getgradientnopadding(nn.Module):
    def __init__(self):
        super(Getgradientnopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class Filter_UNet(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, in_channel=1, num_classes=1, channel=8):
        super().__init__()

        self.get_g_nopadding = Getgradientnopadding()
        # encoding
        self.conv1 = encoding_block(in_channel + 1, channel, kernel_size=5, padding=0)
        self.maxpool1 = nn.Conv2d(channel, channel, kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = encoding_block(channel, 2 * channel, kernel_size=5, padding=0)
        self.maxpool2 = nn.Conv2d(2 * channel, 2 * channel, kernel_size=(2, 2), stride=(2, 2))

        self.center = encoding_block(2 * channel, 4 * channel)

        self.decode2 = decoding_block(4 * channel, 2 * channel, upsampling=True)
        self.decode1 = decoding_block(2 * channel, channel, upsampling=False)

        # final
        self.decode0 = nn.Conv2d(channel, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.final = nn.Conv2d(num_classes + 1, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, input):
        grad = self.get_g_nopadding(input)

        conv1 = self.conv1(torch.concatenate((input, grad), dim=1))
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        # center
        center = self.center(maxpool2)

        decode2 = self.decode2(conv2, center)
        decode1 = self.decode1(conv1, decode2)

        # final
        decode0 = self.decode0(decode1)
        final = self.final(torch.concatenate((decode0, input), dim=1))

        return final


