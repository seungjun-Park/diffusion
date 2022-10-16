from typing import Tuple, Optional

import math
import warnings

import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models.utils import conv

class DiffusionCompression(nn.Module):
    def __init__(self, N, M, entropy_bottleneck_channels,   # for compression
                 noise_model: UNetModel, n_steps: int,        # for diffusion
                 discretize: str = "uniform", eta: float = 0):
        super(DiffusionCompression, self).__init__()

        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)
        self.lamda = torch.rand(0, 1)

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

        self.noise_model = noise_model

        self.n_steps = model.n_steps

    def forward(self, x):
        y0 = self.compression_model.g_a(x)
        z = self.compression_model.h_a(y0)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        batch_size = y0.shape[0]
        t = torch.randint(0, self.n_step, (batch_size, ), device=y0.device, dtype=torch.long)

        yt = self.q_sample(y0, t)

        return yt, z_hat

