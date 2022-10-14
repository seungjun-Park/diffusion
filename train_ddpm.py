from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import lab, tracker, experiment, monit

from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet

class Configs(BaseConfigs):
    device: torch.device = DeviceConfigs()
    eps_model: UNet
    diffusion: DenoiseDiffusion

    image_channels: int = 3
    image_size: int = 32

    n_channels: int = 64
    channel_multipliers: List[int] = [1, 2, 3, 4]

    is_attention: List[int] = [False, False, False, True]

    n_steps: int = 1_000
    batch_size: int = 64
    n_samples: int = 16
    learning_rate: float = 23-5
    epochs: int = 1_000

    dataset: torch.utils.data.Dataset
    data_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Adam

    def init(self):
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

        tracker.set_image("sample", True)

    def sample(self):
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            for t_ in monit.iterate('Sample', self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            tracker.save('sample', x)

    def train(self):
        for data in monit.iterate('Train', self.data_loader):
            tracker.add_global_step()
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()
            tracker.save('loss', loss)

    def run(self):
        for _ in monit.loop(self.epochs):
            self.train()
            self.sample()
            tracker.new_line()
            experiment.save_checkpoint()

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, image_size: int):
        super(CelebADataset, self).__init__()

        folder = lab.get_data_path() / 'celebA'
        self._files = [p for p in folder.glob(f'***/*.jpg')]
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        img = Image.open(self._files[index])
        return self._transform(img)

@option(Configs.dataset, 'CelebA')
def celeb_dataset(c: Configs):
    return CelebADataset(c.image_size)

class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super(MNISTDataset, self).__init__(str(lab.get_data_path()), train=True, transform=transform)

    def __getitem__(self, item):
        return super(MNISTDataset, self).__getitem__(item)[0]

@option(Configs.dataset, 'MNIST')
def minist_dataset(c: Configs):
    return MNISTDataset(c.image_size)

def main():
    experiment.create(name='diffuse', writers={'screen', 'labml'})
    configs = Configs()
    experiment.configs(configs, {
        'dataset': 'CelebA',    # 'MNIST'
        'image_channels': 3,    # 1
        'epochs': 100,          # 5
    })

    configs.init()
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    with experiment.start():
        configs.run()

if __name__ == "__main__":
    main()