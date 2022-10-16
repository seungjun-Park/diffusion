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
from diffusion_compression import DiffusionCompression
from torch.utils.tensorboard import SummaryWriter
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel


def run_inference(device="cpu", test_dir="test", model_dir=""):
    return


def run_training(
        device="cpu",
        train_dir="train",
        save_dir="model_params/",
        n_epochs=100,
        batch_size=5,
        lr=5e-10,
        diff_pth="",
        ckpt_interval=1
    ):

    preprocessing = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
    ])

    print("Using device: {}".format(device))

    save_pth_dir = save_dir + "{}/{}/".format(batch_size, lr)
    print("model params will be saved to: {}".format(save_pth_dir))

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=preprocessing)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_size = len(train_ds)
    print("Number of train samples: ", train_size)


    # Define Model
    noise_model = UNetModel(in_channels=3, out_channels=3, channels=128, n_res_blocks=2,
                            attention_levels=[1, 4], channel_multipliers=[1, 1, 2, 2, 4, 4], n_heads=2) # nheads 모름
    model = DiffusionCompression(N=128, M=256, entropy_bottleneck_channels=64, noise_model=noise_model,
                                 n_steps=100, linear_start=1e-4, linear_end=2e-2).to(device)


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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = {
        "device": device,
        "train_dir": "train",
        "save_dir": "model_params/",
        "n_epochs": 100,
        "batch_size": 5,
        "lr": 5e-10,
        "diff_pth": "",
        "ckpt_interval": 1
    }
    run_training(**configs)