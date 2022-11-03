import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os, time, logging

from compressai.zoo import image_models
from modules.image_dataset import CustomDataset
from modules.ddim.unet import UNet
from modules.ddim.losses import noise_estimation_loss
from modules.ddim.ema import EMAHelper

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def train_model(path,
                betas,
                ch, out_ch, ch_mult,
                num_res_blocks,
                attn_resolutions,
                dropout,
                in_channels,
                image_size,
                resamp_with_conv,
                num_timesteps,
                num_diffusion_timesteps,
                model_type='bayesian',
                use_ema=False,
                resume_training=False,
                ema_rate=0.999,
                epochs=100,
                lr=0.00002,
                weight_decay=0.000,
                beta1=0.9,
                amsgrad=False,
                eps=0.00000001,
                snapshot_freq=10,
                device=None
                ):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = CustomDataset("./data/phases", transform=train_transforms)
    test_dataset = CustomDataset("./test/phases", transform=test_transforms)

    #tb_logger = tb.SummaryWriter(path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models['mbt2018-mean'](quality=3)
    net = net.to(device)

    checkpoint = torch.load("./checkpoint.pth.tar", map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    if not net.update(force=True):
        raise RuntimeError(f'Can not update CDF!')

    eps_model = UNet(ch, out_ch, ch_mult,
         num_res_blocks,
         attn_resolutions,
         dropout,
         in_channels,
         image_size,
         resamp_with_conv,
         num_diffusion_timesteps,
         model_type=model_type)
    eps_model = eps_model.to(device)

    optimizer = optim.Adam(eps_model.parameters(), lr=lr, weight_decay=weight_decay,
                           betas=(beta1, 0.999), amsgrad=amsgrad, eps=eps)

    if use_ema:
        ema_helper = EMAHelper(mu=ema_rate)
        ema_helper.register(eps_model)
    else:
        ema_helper = None

    start_epoch, step = 0, 0
    if resume_training:
        states = torch.load("./ckpt.pth")
        eps_model.load_state_dict(states[0])

        states[1]["param_groups"][0]["eps"] = eps
        optimizer.load_state_dict(states[1])
        start_epoch = states[2]
        step = states[3]
        if use_ema:
            ema_helper.load_state_dict(states[4])

    for epoch in range(start_epoch, epochs):
        data_start = time.time()
        data_time = 0
        step = 0
        mean_loss= 0
        for i, x in enumerate(train_dataloader):
            data_time += time.time() - data_start
            eps_model.train()
            step += 1

            x = x.to(device)
            n = x.size(0)
            e = torch.randn_like(x)
            b = betas

            # antithetic sampling
            t = torch.randint(
                low=0, high=num_timesteps, size=(n // 2 + 1,)
            ).to(device)
            t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
            loss = noise_estimation_loss(eps_model, x, t, e, b)

            #tb_logger.add_scalar("loss", loss, global_step=step)

            logging.info(
                f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}"
            )

            mean_loss += loss.item()

            print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}")

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    eps_model.parameters(), 1.0
                )
            except Exception:
                pass
            optimizer.step()

            if use_ema:
                ema_helper.update(eps_model)

            if step % snapshot_freq == 0 or step == 1:
                states = [
                    eps_model.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if use_ema:
                    states.append(ema_helper.state_dict())

                # torch.save(
                #     states,
                #     os.path.join(path, "ckpt_{}.pth".format(step)),
                # )
                torch.save(states, os.path.join(path, "ckpt.pth.tar"))

            data_start = time.time()


def sample_image(x,
                 ch, out_ch, ch_mult,
                 num_res_blocks,
                 attn_resolutions,
                 dropout,
                 in_channels,
                 image_size,
                 resamp_with_conv,
                 num_diffusion_timesteps,
                 betas,
                 num_timesteps,
                 device,
                 eta=0.0,
                 timesteps=1000,
                 model_type='bayesian',
                 skip=1,
                 last=True,
                 sample_type='generalized',
                 skip_type='uniform'):
    x = torch.randn_like(x)
    x.to(device)

    model = UNet(ch, out_ch, ch_mult,
         num_res_blocks,
         attn_resolutions,
         dropout,
         in_channels,
         image_size,
         resamp_with_conv,
         num_diffusion_timesteps,
         model_type=model_type)
    model = model.to(device)
    checkpoint = torch.load("./ckpt.pth.tar", map_location=device)
    model.load_state_dict(checkpoint[0])

    if sample_type == "generalized":
        if skip_type == "uniform":
            skip = num_timesteps // timesteps
            seq = range(0, num_timesteps, skip)
        elif skip_type == "quad":
            seq = (
                    np.linspace(
                        0, np.sqrt(num_timesteps * 0.8), timesteps
                    )
                    ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        from modules.ddim.denoising import generalized_steps

        xs = generalized_steps(x, seq, model, betas, eta=eta, last=True)
        x = xs
    elif sample_type == "ddpm_noisy":
        if skip_type == "uniform":
            skip = num_timesteps // timesteps
            seq = range(0, num_timesteps, skip)
        elif skip_type == "quad":
            seq = (
                    np.linspace(
                        0, np.sqrt(num_timesteps * 0.8), timesteps
                    )
                    ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        from modules.ddim.denoising import ddpm_steps

        x = ddpm_steps(x, seq, model, betas)
    else:
        raise NotImplementedError
    if last:
        x = x[0][-1]
    return x


def compress(x, device):
    net = image_models['mbt2018-mean'](quality=3)
    net = net.to(device)

    checkpoint = torch.load("./checkpoint.pth.tar", map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    if not net.update(force=True):
        raise RuntimeError(f'Can not update CDF!')

    y = net.g_a(x)
    z = net.h_a(y)

    z_strings = net.entropy_bottleneck.compress(z)
    z_hat = net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    gaussian_params = net.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = net.gaussian_conditional.build_indexes(scales_hat)
    y_strings = net.gaussian_conditional.compress(y, indexes, means=means_hat)
    return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

def decompress(strings, shape, betas, num_diffusion_timesteps, num_timestpes, device):
    net = image_models['mbt2018-mean'](quality=3)
    net = net.to(device)

    checkpoint = torch.load("./checkpoint.pth.tar", map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    if not net.update(force=True):
        raise RuntimeError(f'Can not update CDF!')

    assert isinstance(strings, list) and len(strings) == 2
    z_hat = net.entropy_bottleneck.decompress(strings[1], shape)
    gaussian_params = net.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = net.gaussian_conditional.build_indexes(scales_hat)
    y_hat = net.gaussian_conditional.decompress(
         strings[0], indexes, means=means_hat
    )
    y_noise = sample_image(y_hat,
                         betas=betas, in_channels=192, out_ch=192, ch=128,
                         ch_mult=tuple([1, 1, 2, 2, 4, 4, ]),
                         num_res_blocks=2, attn_resolutions=[16, ], dropout=0.0,
                         image_size=2048, resamp_with_conv=True,
                         num_diffusion_timesteps=num_diffusion_timesteps, skip=20, num_timesteps=num_timestpes, device=device)
    y_noise = y_noise.to(device)

    x_hat = net.g_s(y_noise).clamp_(0, 1)
    return {"x_hat": x_hat}

def image_recon(path, betas, num_timesteps, num_diffusion_timsteps, device):
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CustomDataset("./data/phases", transform=data_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    for i, x in enumerate(dataloader):
        x = x.to(device)

        state_dict = compress(x, device)
        x_hat = decompress(state_dict['strings'], state_dict['shape'], betas, num_diffusion_timsteps, num_timesteps, device)['x_hat']
        toImage = transforms.ToPILImage()
        x_hat = x_hat[0]
        x_hat = toImage(x_hat)

        x_hat.save(f"{path}/recon/{i}.png")

def main(path,
         beta_schedule,
         beta_start,
         beta_end,
         num_diffusion_timesteps,
         model_var_type,
         train=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    betas = get_beta_schedule(
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        num_diffusion_timesteps=num_diffusion_timesteps
    )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]

    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
    )
    posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    if model_var_type == "fixedlarge":
        logvar = betas.log()
        # torch.cat(
        # [posterior_variance[1:2], betas[1:]], dim=0).log()
    elif model_var_type == "fixedsmall":
        logvar = posterior_variance.clamp(min=1e-20).log()

    if train:
        train_model(path=path, betas=betas, in_channels=1, out_ch=1, ch=128, ch_mult=tuple([1, 1, 2, 2, 4, 4,]),
                    num_res_blocks=2, attn_resolutions=[16, ], dropout=0.0, ema_rate=0.999, use_ema=True, image_size=2048, resamp_with_conv=True, num_timesteps=num_timesteps,
                    num_diffusion_timesteps=num_diffusion_timesteps, device=device)

    else:
        image_recon(path, betas, num_timesteps, num_diffusion_timesteps, device)

if __name__ == '__main__':
    main(path='./',
         beta_schedule='linear', beta_start=0.0001, beta_end=0.02,
         num_diffusion_timesteps=1000, model_var_type='fixedsmall', train=True)



