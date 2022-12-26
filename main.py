import argparse

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

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

if __name__ == "__main__":
    p = get_parser()
    arg = p.parse_args()
    p = Trainer.add_argparse_args(p)
    opt, unknown = p.parse_known_args()
    config = [OmegaConf.load(cfg) for cfg in arg.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*config, cli)

    print(config.diffusion.params.linear_start)