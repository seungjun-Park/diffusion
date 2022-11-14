import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from modules.ema import LitEma
from modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like

#from models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL

