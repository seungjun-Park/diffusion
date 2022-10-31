from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models.utils import conv, deconv
from labml import monit
import lpips


class LDCM(nn.Module):
    def __init__(self, config):
        super(LDCM, self).__init__()

        self.config = config
        self.device = torch.device(config['device'])

        self.lamda = config['lambda']
        self.lo = config['lo']
        in_channel = config['in_channel']
        out_channel = config['out channel']
        N = config['N']
        M = config['M']

        self.entropy_bottleneck = EntropyBottleneck(config['entropy_bottleneck_channels'])

        self.g_a = nn.Sequential(
            conv(in_channel, N),
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

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, out_channel),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )


        self.n_steps = config['n_steps']
        self.eps_model = UNet(config['UNet'])

        if discretize == "uniform":
            c = self.n_steps // step_range
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

    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor, *,
                               temperature: float,
                               repeat_noise: bool):
        alpha = self.alpha[index]
        alpha_prev = self.alpha_prev[index]
        sigma = self.sigma[index]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha[index]
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        if sigma == 0:
            noise = 0.

        elif repeat_noise:
            noise = torch.randn(1, *x.shape[1:], device=x.device)

        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature

        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0

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

        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x,
                                                      temperature=temperature,
                                                      repeat_noise=repeat_noise)

        return x_prev, pred_x0, e_t

    def loss(self,
             x: torch.Tensor, noise: torch.Tensor = None,
             repeat_noise: bool = False,
             temperature: float = 1.,
             uncond_scale: float = 1.,
             uncond_cond: Optional[torch.Tensor] = None,
             skip_steps: int = 0):

        y0 = self.g_a(x)
        z = self.h_a(y0)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        y_noise = torch.randn_like(y0)
        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in monit.enum('Sample', time_steps):
            index = len(time_steps) - i - 1
            ts = x.new_full((y0.shape[0],), step, dtype=torch.long)
            y_prev, pred_y0, e_t = self.p_sample(y_noise, None, ts, step, index=index,
                                                 repeat_noise=repeat_noise,
                                                 temperature=temperature,
                                                 uncond_scale=uncond_scale,
                                                 uncond_cond=uncond_cond)

        l1_loss = nn.L1Loss()
        if noise is None:
            noise = torch.randn_like(y0)
        l1_loss(noise, e_t)

        d = lpips.LPIPS(net='alex')
        d = d.forward(pred_y0, y0)

        return (1. - self.lo) * l1_loss + self.lo * d -  self.lamda * torch.mean(torch.log2(z_likelihoods))

    @torch.no_grad()
    def compress_test(self, x: torch.Tensor,
                repeat_noise: bool = False,
                temperature: float = 1.,
                uncond_scale: float = 1.,
                uncond_cond: Optional[torch.Tensor] = None,
                skip_steps: int = 0):

        with torch.no_grad():
            y0 = self.g_a(x)
            z = self.h_a(y0)

            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

            y_noise = torch.randn_like(y0)
            time_steps = np.flip(self.time_steps)[skip_steps:]

            for i, step in monit.enum('Sample', time_steps):
                index = len(time_steps) - i - 1
                ts = x.new_full((y0.shape[0],), step, dtype=torch.long)
                y_noise, pred_y0, e_t = self.p_sample(y_noise, None, ts, step, index=index,
                                                      repeat_noise=repeat_noise,
                                                      temperature=temperature,
                                                      uncond_scale=uncond_scale,
                                                      uncond_cond=uncond_cond)
            x_hat = self.g_s(pred_y0).clamp(0, 1)
            return x_hat
