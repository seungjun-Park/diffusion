import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from compressai.zoo import image_models
from modules.image_dataset import CustomDataset
from models.diffusion.ddpm import DDPM

def main(path):
    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = CustomDataset("./data/phases", transform=train_transforms)
    test_dataset = CustomDataset("./test/phases", transform=test_transforms)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    diffusion_model = DDPM(device=device)

    net = image_models['mbt2018-mean'](quality=3)
    net = net.to(device)

    checkpoint = torch.load("./checkpoint.pth.tar", map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    if not net.update(force=True):
        raise RuntimeError(f'Can not update CDF!')

    for i, x in enumerate(train_dataloader):
        x = x.to(device)
        compress = net(x)
        y = compress['diffusion']['y']
        z = compress['diffusion']['z']

        loss = diffusion_model.loss(x0=y)
        loss.backward()

