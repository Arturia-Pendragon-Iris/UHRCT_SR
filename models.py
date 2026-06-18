import torch
import torch.nn as nn


class RED(nn.Module):
    def __init__(self, in_ch=1, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch * 2, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch * 4, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch * 2, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))

        out = self.tconv1(out)
        out = out + residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out = out + residual_2
        out = self.tconv4(self.relu(out))
        return self.tconv5(self.relu(out))


class Dual_unet(nn.Module):
    def __init__(self, image_channels=1, structure_channels=1, base_channels=32, out_channels=1):
        super().__init__()
        self.image_channels = image_channels
        self.structure_channels = structure_channels

        self.resnet_img = RED(in_ch=image_channels, out_ch=base_channels)
        self.resnet_vessel = RED(in_ch=structure_channels, out_ch=base_channels)

        self.img_trans = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

        self.vessel_trans = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

        self.img_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.vessel_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels, structure_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, vessel):
        """
        Args:
            x: CT image input. In diffusion training this can be the concatenation
               of LRCT condition, noisy HRCT, and optional noise-level channels.
            vessel: Frangi-enhanced structural prior aligned with x.
        """
        out_img = self.img_trans(self.resnet_img(x))
        out_vessel = self.vessel_trans(self.resnet_vessel(vessel))

        image_output = self.img_fusion(torch.cat((out_img, out_vessel), dim=1))
        structure_output = self.vessel_fusion(torch.cat((out_vessel, out_img), dim=1))
        return image_output, structure_output


class DSSPNet(Dual_unet):
    pass
