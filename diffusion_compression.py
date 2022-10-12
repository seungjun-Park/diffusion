from typing import Tuple, Optional

import math
import warnings

import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model

from compressai.models.utils import conv, deconv, update_registered_buffers

from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion

from ddim import DDIM
from models import MeanScaleHyperprior

class DiffusionCompression(nn.Module):
    def __init__(self, diffusion_model: DDIM, compression_model: MeanScaleHyperprior, n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
        super(DiffusionCompression, self).__init__()

        self.compression_model = compression_model

    def forward(self, x):
        y = self.compression_model.g_a(x)
        x_t = self.q_sample(x, self.n_steps)
        z = self.compression_model.h_a(y)
        z_hat, z_likelihoods = self.compression_model.entropy_bottleneck(z)
