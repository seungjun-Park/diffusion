import os
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

from model.diffusion.ddpm import DDPM
from labml_nn.diffusion.ddpm.unet import UNet
from typing import List

def run_inference(device="cpu", test_dir="test", model_dir=""):
    return


def run_training(
        device: torch.device,
        train_dir="data",
        save_dir="model_params/",
        n_epochs=100,
        batch_size=5,
        lr=5e-10,
        diff_pth="",
        ckpt_interval=1
    ):

    preprocessing = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print("Using device: {}".format(device))

    save_pth_dir = save_dir + "{}/{}/".format(batch_size, lr)
    print("model params will be saved to: {}".format(save_pth_dir))

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=preprocessing)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_size = len(train_ds)
    print("Number of train samples: ", train_size)

    image_channels: int = 1
    image_size: int = 256

    n_channels: int = 64
    channel_multipliers: List[int] = [1, 2, 3, 4]

    is_attention: List[int] = [False, False, False, True]

    n_steps: int = 1_000
    batch_size: int = 64
    n_samples: int = 16


    # Define Model
    unet_model = UNet(image_channels=image_channels,
            n_channels=n_channels,
            ch_mults=channel_multipliers,
            is_attn=is_attention).to(device)

    model = DDPM(eps_model=unet_model,
                 n_steps=n_steps,
                 device=torch.device)


    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    writer = SummaryWriter(comment="train")

    # Train the model
    for epoch in range(n_epochs):

        print("Epoch: {}".format(epoch + 1))

        for step, (x, labels) in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()

            x = x.to(device)

            loss = model.loss(x)

            loss.backward()

            optimizer.step()

            writer.add_scalar("Batch/Train-loss", loss.item(), (step + 1) + int(train_size / batch_size) * epoch)

        # Save model
        if (epoch + 1) % ckpt_interval:
            torch.save(model.state_dict(), save_pth_dir + "diffusion_{}.pth".format(epoch + 1))


    # x = torch.randn([n_samples, image_channels, image_size, image_size],
    #                 device=torch.device(device))
    #
    # for t_ in range(0, n_steps):
    #     t = n_steps - t_ - 1
    #     x = model.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
    #
    # tracker.save('sample', x)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = {
        "device": device,
        "train_dir": "data",
        "save_dir": "model_params/",
        "n_epochs": 100,
        "batch_size": 5,
        "lr": 5e-10,
        "diff_pth": "",
        "ckpt_interval": 1
    }
    run_training(**configs)