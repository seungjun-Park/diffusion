from typing import Tuple, Optional

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models.utils import conv

from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel

class DiffusionCompression(nn.Module):
    def __init__(self, N, M, entropy_bottleneck_channels,   # for compression
                 noise_model: UNetModel,
                 n_steps: int, linear_start: float, linear_end: float, latent_sacling_factor,
                 discretize: str = "uniform", eta: float = 0):
        super(DiffusionCompression, self).__init__()

        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        self.lamda: int = 1.0
        self.lo: int = 0.9
        self.learning_rate: int

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.n_steps = n_steps
        self.noise_model = noise_model

        if discretize == "uniform":
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) + 1

        elif discretize == "quad":
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * 0.8))) ** 2).astype(int) + 1

        else:
            raise NotImplementedError(discretize)

        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.alpha = self.alpha_bar[self.time_steps].clone().to(torch.float32)
        self.alpha_sqrt = torch.sqrt(self.alpha)
        self.alpha_prev = torch.cat([alpha_bar[0 : 1], alpha_bar[self.time_steps[ : -1]]])
        self.sigma = (eta * ((1 - self.alpha_prev) / (1 - self.alpha) * (1 - self.alpha / self.alpha_prev)) ** .5)
        self.sqrt_one_minus_alpha = (1. - self.alpha) ** .5

    def device(self):
        return next(iter(self.noise_model.parameters())).device()

    def get_noise(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, *,
                  uncond_scale: float, uncond_cond: Optional[torch.Tensor] = None):

        if uncond_cond is None or uncond_scale == 1.:
            return self.noise_model(x, t, c)

        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)

        c_in = torch.cat([uncond_cond, c])

        e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)

        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        return e_t

    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)

        return self.alpha_sqrt[index] * x0 + self.sqrt_one_minus_alpha[index] * noise

    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int, *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: Optional[torch.Tensor] = None):
        e_t = self.get_noise(x, t, c,
                             uncond_scale=uncond_scale,
                             uncond_cond=uncond_cond)


    def forward(self, x):
        y0 = self.compression_model.g_a(x)
        z = self.compression_model.h_a(y0)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        batch_size = y0.shape[0]
        t = torch.randint(0, self.n_step, (batch_size, ), device=y0.device, dtype=torch.long)

        yt = self.q_sample(y0, t)

        return

