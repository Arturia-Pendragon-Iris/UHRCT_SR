import torch
import torch.nn as nn


class RED(nn.Module):
    def __init__(self, out_ch=64):
        super(RED, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch * 2, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch * 4, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch * 2, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        return out


class Dual_unet(nn.Module):
    def __init__(self):
        super(Dual_unet, self).__init__()
        self.resnet_img = RED(out_ch=32)
        self.resnet_vessel = RED(out_ch=32)

        self.img_trans = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        )

        self.vessel_trans = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        )

        self.img_fusion = nn.Sequential(
            nn.Conv2d(1 + 1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        )

        self.vessel_fusion = nn.Sequential(
            nn.Conv2d(1 + 1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        )



    def forward(self, x, vessel):
        """
        :param x: the CT slice
        :param vessel: the filtered result of the CT slice
        """
        # encoder
        out_img = self.resnet_img(x)
        out_vessel = self.resnet_vessel(vessel)

        out_img = self.img_trans(out_img)
        out_vessel = self.vessel_trans(out_vessel)

        out_img = self.img_fusion(torch.concat((out_img, out_vessel), dim=1))
        out_img = self.vessel_fusion(torch.concat((out_vessel, out_img), dim=1))

        return out_img, out_vessel