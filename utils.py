import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GetSobel(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
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


class GetLaplace(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        self.weight = nn.Parameter(torch.FloatTensor(kernel).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i:i + 1]
            x_list.append(F.conv2d(x_i, self.weight.to(x.device), padding=1))
        return torch.cat(x_list, dim=1)


class GetHighPass(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]
        kernel_h = [[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]]
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


class FrangiFilter2D(nn.Module):
    def __init__(self, sigmas=(0.5, 1.0, 1.5, 2.0), beta=0.5, gamma=None, black_ridges=False):
        super().__init__()
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.beta = float(beta)
        self.gamma = gamma
        self.black_ridges = black_ridges

    @staticmethod
    def _gaussian_derivative_kernels(sigma, device, dtype):
        radius = max(int(math.ceil(3.0 * sigma)), 1)
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        gaussian = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
        gaussian = gaussian / gaussian.sum().clamp_min(1e-12)
        first = -x / (sigma ** 2) * gaussian
        second = ((x ** 2 - sigma ** 2) / (sigma ** 4)) * gaussian
        return gaussian, first, second

    @staticmethod
    def _depthwise_conv(input_tensor, kernel):
        channels = input_tensor.shape[1]
        kernel = kernel.expand(channels, 1, kernel.shape[-2], kernel.shape[-1])
        pad_y = kernel.shape[-2] // 2
        pad_x = kernel.shape[-1] // 2
        input_tensor = F.pad(input_tensor, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
        return F.conv2d(input_tensor, kernel, groups=channels)

    def _hessian(self, x, sigma):
        gaussian, first, second = self._gaussian_derivative_kernels(sigma, x.device, x.dtype)
        kxx = torch.outer(gaussian, second).view(1, 1, gaussian.numel(), second.numel())
        kyy = torch.outer(second, gaussian).view(1, 1, second.numel(), gaussian.numel())
        kxy = torch.outer(first, first).view(1, 1, first.numel(), first.numel())
        scale = sigma ** 2
        return (
            scale * self._depthwise_conv(x, kxx),
            scale * self._depthwise_conv(x, kxy),
            scale * self._depthwise_conv(x, kyy),
        )

    def forward(self, x):
        responses = []
        eps = torch.finfo(x.dtype).eps
        for sigma in self.sigmas:
            dxx, dxy, dyy = self._hessian(x, sigma)
            tmp = torch.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2 + eps)
            lambda_a = 0.5 * (dxx + dyy + tmp)
            lambda_b = 0.5 * (dxx + dyy - tmp)
            swap = torch.abs(lambda_a) > torch.abs(lambda_b)
            lambda1 = torch.where(swap, lambda_b, lambda_a)
            lambda2 = torch.where(swap, lambda_a, lambda_b)

            rb = torch.abs(lambda1) / torch.abs(lambda2).clamp_min(eps)
            s = torch.sqrt(lambda1 ** 2 + lambda2 ** 2 + eps)
            c = self.gamma
            if c is None:
                c = torch.amax(s.detach(), dim=(-2, -1), keepdim=True).clamp_min(eps) * 0.5

            response = torch.exp(-(rb ** 2) / (2.0 * self.beta ** 2))
            response = response * (1.0 - torch.exp(-(s ** 2) / (2.0 * c ** 2)))
            if self.black_ridges:
                response = torch.where(lambda2 < 0, torch.zeros_like(response), response)
            else:
                response = torch.where(lambda2 > 0, torch.zeros_like(response), response)
            responses.append(response)

        vesselness = torch.stack(responses, dim=0).amax(dim=0)
        scale = torch.amax(vesselness.detach(), dim=(-2, -1), keepdim=True).clamp_min(eps)
        return vesselness / scale


def ln_loss(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    loss_2 = nn.MSELoss()
    return loss_1(prediction_results, ground_truth) + 5 * loss_2(prediction_results, ground_truth)


def frangi_loss(prediction_results, ground_truth, frangi_filter=None):
    loss_1 = nn.L1Loss()
    frangi_filter = frangi_filter or FrangiFilter2D()
    frangi_filter = frangi_filter.to(prediction_results.device)
    return loss_1(frangi_filter(prediction_results), frangi_filter(ground_truth))


def sharp_loss(prediction_results, ground_truth):
    return frangi_loss(prediction_results, ground_truth)


def grad_loss_simple(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    grad_pre = prediction_results[:, :, 1:, 1:, 1:] - prediction_results[:, :, :-1, :-1, :-1]
    grad_gt = ground_truth[:, :, 1:, 1:, 1:] - ground_truth[:, :, :-1, :-1, :-1]
    return loss_1(grad_pre, grad_gt)


def structure_loss(image_output, structure_output, ground_truth, frangi_filter=None):
    loss_1 = nn.L1Loss()
    frangi_filter = frangi_filter or FrangiFilter2D()
    frangi_filter = frangi_filter.to(image_output.device)
    target_structure = frangi_filter(ground_truth)
    return loss_1(frangi_filter(image_output), target_structure) + loss_1(structure_output, target_structure)


def reconstruction_loss(prediction_results, ground_truth, lamb=0.5):
    ln_img = ln_loss(prediction_results, ground_truth)
    ln_vessel = frangi_loss(prediction_results, ground_truth)
    return ln_img + lamb * ln_vessel


def dual_stream_reconstruction_loss(image_output, structure_output, ground_truth, lamb=0.5):
    return ln_loss(image_output, ground_truth) + lamb * structure_loss(image_output, structure_output, ground_truth)


def smse(prediction_results, ground_truth, frangi_filter=None):
    frangi_filter = frangi_filter or FrangiFilter2D()
    frangi_filter = frangi_filter.to(prediction_results.device)
    return F.mse_loss(frangi_filter(prediction_results), frangi_filter(ground_truth))
