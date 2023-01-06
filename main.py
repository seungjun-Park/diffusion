import argparse
import os, sys, glob, datetime

import torch
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

class Wrapper(pl.LightningModule):
    def __int__(self, config):
        super().__int__()
        self.compression_model = instantiate_from_config(config.model)
        self.criterion = instantiate_from_config(config.criterion)
        self.learning_rate = config.learning_rate
        self.aux_learning_rate = config.aux_learning_rate
        self.automatic_optimization = False
    def training_Step(self, batch, batch_idx):
        x = batch
        out = self.compression_model(x)
        opt = self.optimizers()
        opt[0].zero_grad()

        opt[0].step()

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