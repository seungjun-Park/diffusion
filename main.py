import argparse

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

from util import get_module, instantiate_from_config

import matplotlib.pyplot as plt

def get_parser(**parser_kwargs):

    arg_parser = argparse.ArgumentParser(**parser_kwargs)
    arg_parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        default=list()
    )

    return arg_parser

def Test(*args, **kwargs):
    return

if __name__ == "__main__":
    parsers = get_parser()
    parsers = Trainer.add_argparse_args(parsers)

    opt, unknown = parsers.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config['accelerator'] = 'ddp'

    Test(*lightning_config, **config)

    trainer_opt = argparse.Namespace(**trainer_config)

    model = instantiate_from_config(config.model)

    trainer_kwargs = dict()

    data = get_module(config.data)
    data.prepare_data()
    data.setup()

    t =
