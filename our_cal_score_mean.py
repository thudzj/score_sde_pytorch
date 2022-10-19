from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import wrapper
from fid_utils import get_fid
import argparse

parser = argparse.ArgumentParser(description='Eval')
parser.add_argument('--batch-size', default=1024, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--n-estimates', default=1, type=int)


def main():
    args = parser.parse_args()

    sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
    if sde.lower() == 'vesde':
        from configs.ve import cifar10_ncsnpp_continuous as configs
        ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
        config = configs.get_config()  
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    elif sde.lower() == 'vpsde':
        from configs.vp import cifar10_ddpmpp_continuous as configs  
        ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
        config = configs.get_config()
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif sde.lower() == 'subvpsde':
        from configs.subvp import cifar10_ddpmpp_continuous as configs
        ckpt_filename = "exp/subvp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
        config = configs.get_config()
        sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3

    batch_size = args.batch_size
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size

    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    img_size = config.data.image_size
    channels = config.data.num_channels

    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=config.training.continuous)
    # Build data iterators
    train_ds, _, _ = datasets.get_dataset(config,
                                        uniform_dequantization=config.data.uniform_dequantization,
                                        evaluation=True)
    train_iter = iter(train_ds)

    with torch.no_grad():
        timesteps = torch.linspace(sde.T, sampling_eps, sde.N, device=config.device)

        score_sum = torch.zeros(sde.N, channels, img_size, img_size).to(config.device)
        score_normsqr_sum = torch.zeros(sde.N).to(config.device)
        n_data = 0
        
        while 1:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            x = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
            x = x.permute(0, 3, 1, 2)
            x = scaler(x)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(x.shape[0], device=t.device) * t 
                mean, std = sde.marginal_prob(x, vec_t)
                
                for _ in range(args.n_estimates):
                    perturbed_data = mean + std[:, None, None, None] * torch.randn_like(x)
                    score = score_fn(perturbed_data, vec_t)
                    score_sum[i] += score.sum(0)
                    score_normsqr_sum[i] += (score.flatten(1).norm(dim=1) ** 2).sum(0)
            
            n_data += args.n_estimates * x.shape[0]
            print(n_data / args.n_estimates)

            score_mean = score_sum / n_data
            score_mean_normsqr = score_mean.flatten(1).norm(dim=1) ** 2
            score_normsqr_mean = score_normsqr_sum / n_data
            ratio = (score_mean_normsqr / score_normsqr_mean).sqrt()
            torch.save(score_mean.cpu(), "mean_scores/{}_".format(n_data // args.n_estimates) + ckpt_filename.replace("/", "_"))

if __name__ == '__main__':
    main()