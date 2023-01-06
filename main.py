import argparse
import os, sys, glob, datetime, math

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from packaging import version

from omegaconf import OmegaConf
from torch import optim
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything

from util import instantiate_from_config, str2bool

import matplotlib.pyplot as plt

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help='path to base configs. Loaded from left-to-right. '
              'Parameters can be oeverwritten or added with command-line options of the form "--key value".',
        default=list(),
    )

    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything'
    )

    parser.add_argument(
        '-l',
        '--logdir',
        type=str,
        default='logs',
        help='directory for logging dat shift',
    )

    parser.add_argument(
        '--scale_lr',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='scale base-lr by ngpu * batch_size * n_accumulate',
    )

    return parser

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out

class Wrapper(pl.LightningModule):
    def __int__(self, config):
        super().__int__()
        self.compression_model = instantiate_from_config(config.model)
        self.criterion = instantiate_from_config(config.criterion)
        self.learning_rate = config.learning_rate
        self.aux_learning_rate = config.aux_learning_rate
        self.automatic_optimization = False
        self.clip_max_norm = 1.0
    def training_Step(self, batch, batch_idx):
        optimizer, aux_optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        x = batch
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = self.compression_model(x)

        out_criterion = self.criterion(out, x)
        out_criterion["loss"].backward()
        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()


        lr_scheduler.step()

    def configure_optimizers(self):
        parameters = {
            n
            for n, p in self.compression_model.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        aux_parameters = {
            n
            for n, p in self.compression_model.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.compression_model.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        optimizer = optim.Adam(
            (params_dict[n] for n in sorted(parameters)),
            lr=self.learning_rate,
        )
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=self.aux_learning_rate,
        )
        return (
            {'optimizer': optimizer,
             'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')},
            {'optimizer': aux_optimizer}
        )


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    parsers = get_parser()
    parsers = Trainer.add_argparse_args(parsers)

    opt, unknown = parsers.parse_known_args()

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # model
    model = instantiate_from_config(config.model)

    trainer = Trainer(accelerator='gpu')
    trainer.fit(model=model, datamodule=data)