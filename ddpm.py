from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data

from labml_nn.diffusion.ddpm.utils import gather

class DDPM:
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model                                       # noise generator

        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)    # beta scheduling
        self.alpha = 1. - self.beta                                     # alpha = 1 - beta

        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.n_step = n_steps

        self.sigma2 = self.beta                                         # variance = beta * I(Idenditity Matrix)

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha) ** .5

        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.rand(xt.shape, device=xt.device)

        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_step, (batch_size, ), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps=noise)

        eps_theta = self.eps_model(xt, t)

        return F.mse_loss(noise, eps_theta)