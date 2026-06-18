from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_filter.model import Filter_UNet
from utils import FrangiFilter2D


def load_filter_operator(path=None, device=None, trainable=False):
    if path is None:
        operator = FrangiFilter2D()
    else:
        operator = Filter_UNet()
        state = torch.load(path, map_location="cpu")
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        operator.load_state_dict(state_dict, strict=False)
    if device is not None:
        operator = operator.to(device)
    operator.train(trainable)
    for parameter in operator.parameters():
        parameter.requires_grad = trainable
    return operator


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        steps=2000,
        sample_steps=100,
        structure_operator=None,
        structure_weight=0.5,
    ):
        super().__init__()
        self.train_steps = steps
        self.sample_steps = sample_steps
        self.model = model
        self.structure_operator = structure_operator or FrangiFilter2D()
        self.structure_weight = structure_weight

        betas = torch.linspace(start=1e-4, end=0.005, steps=self.train_steps)
        alphas = 1.0 - betas
        self.register_buffer("alphas_bar", torch.cumprod(alphas, dim=0))

        sample_betas = torch.linspace(start=1e-4, end=0.1, steps=self.sample_steps)
        sample_alphas = 1.0 - sample_betas
        self.register_buffer("sample_alphas_bar", torch.cumprod(sample_alphas, dim=0))
        self.register_buffer("sample_one_over_sqrt_alphas", torch.sqrt(1.0 / sample_alphas))
        self.register_buffer(
            "sample_betas_over_sqrt_one_minus_alphas_bar",
            sample_betas / torch.sqrt(1.0 - self.sample_alphas_bar),
        )
        self.register_buffer("sample_sigmas", torch.sqrt(sample_betas))

    def _extract(self, values, times, dimension_num):
        batch, *_ = times.shape
        selected_values = torch.gather(values, dim=0, index=times)
        return selected_values.reshape((batch, *[1 for _ in range(dimension_num - 1)]))

    @staticmethod
    def _norm(img):
        return (img - 0.5) * 2.0

    @staticmethod
    def _denorm(img):
        return img / 2.0 + 0.5

    def _structure(self, img):
        return self.structure_operator(torch.clamp(img, 0.0, 1.0))

    def _predict(self, x_t, x_c_model, x_c_image, gamma):
        gamma_channel = gamma.expand(-1, 1, x_t.shape[-2], x_t.shape[-1])
        structure_condition = self._structure(x_c_image)
        model_input = torch.cat((x_t, x_c_model, gamma_channel), dim=1)
        return self.model(model_input, structure_condition)

    @torch.no_grad()
    def sample(self, x_c, x_t=None):
        x_c = torch.clamp(x_c, 0.0, 1.0)
        x_c_norm = self._norm(x_c)
        x_t = torch.randn_like(x_c) if x_t is None else x_t
        batch = x_t.shape[0]
        dimension_num = len(x_t.shape)

        for step in tqdm(reversed(range(self.sample_steps)), total=self.sample_steps, desc="[Sample]", leave=False):
            time = step * x_t.new_ones((batch,), dtype=torch.int64)
            gamma = self._extract(self.sample_alphas_bar, time, dimension_num)
            epsilon, _ = self._predict(x_t, x_c_norm, x_c, torch.sqrt(gamma))

            one_over_sqrt_alpha = self._extract(self.sample_one_over_sqrt_alphas, time, dimension_num)
            beta_over_sqrt_one_minus_alpha_bar = self._extract(
                self.sample_betas_over_sqrt_one_minus_alphas_bar,
                time,
                dimension_num,
            )
            sigma = self._extract(self.sample_sigmas, time, dimension_num)
            z = torch.randn_like(x_t) if step > 0 else 0
            x_t = one_over_sqrt_alpha * (x_t - beta_over_sqrt_one_minus_alpha_bar * epsilon) + sigma * z

        return torch.clamp(self._denorm(x_t), 0.0, 1.0)

    def forward(self, x_0, x_c):
        batch = x_0.shape[0]
        dimension_num = len(x_0.shape)
        x_0 = torch.clamp(x_0, 0.0, 1.0)
        x_c = torch.clamp(x_c, 0.0, 1.0)
        x_0_norm = self._norm(x_0)
        x_c_norm = self._norm(x_c)

        time = torch.randint(low=1, high=self.train_steps, size=(batch,), device=x_0.device)
        gamma_high = self._extract(self.alphas_bar, time - 1, dimension_num)
        gamma_low = self._extract(self.alphas_bar, time, dimension_num)
        gamma = (gamma_high - gamma_low) * torch.rand_like(gamma_high) + gamma_low

        epsilon = torch.randn_like(x_0_norm)
        x_t = torch.sqrt(gamma) * x_0_norm + torch.sqrt(1.0 - gamma) * epsilon
        epsilon_pred, structure_pred = self._predict(x_t, x_c_norm, x_c, torch.sqrt(gamma))

        diffusion_loss = F.mse_loss(epsilon_pred, epsilon)
        x0_pred = (x_t - torch.sqrt(1.0 - gamma) * epsilon_pred) / torch.sqrt(gamma)
        x0_pred = torch.clamp(self._denorm(x0_pred), 0.0, 1.0)
        target_structure = self._structure(x_0).detach()
        structure_loss = F.l1_loss(structure_pred, target_structure)
        structure_loss = structure_loss + F.l1_loss(self._structure(x0_pred), target_structure)
        return diffusion_loss + self.structure_weight * structure_loss
