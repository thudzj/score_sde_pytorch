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
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--method', default='original', type=str)


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
    shape = (batch_size, channels, img_size, img_size)
    predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
    corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
    snr = 0.16 #@param {"type": "number"}
    n_steps =  1#@param {"type": "integer"}
    probability_flow = False #@param {"type": "boolean"}

    if args.method == "original":
        sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                            inverse_scaler, snr, n_steps=n_steps,
                                            probability_flow=probability_flow,
                                            continuous=config.training.continuous,
                                            eps=sampling_eps, device=config.device)
    elif args.method == "sec1.3":
        sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device, 
                                      use_wrapper=True,
                                      calibration=True,
                                      score_mean=None)
    elif args.method == "sec1.4":
        sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device, 
                                      use_wrapper=True,
                                      calibration=False,
                                      score_mean='mean_scores/28672_exp_ve_cifar10_ncsnpp_continuous_checkpoint_24.pth')
    elif args.method == "both":
        sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device, 
                                      use_wrapper=True,
                                      calibration=True,
                                      score_mean='mean_scores/28672_exp_ve_cifar10_ncsnpp_continuous_checkpoint_24.pth')
    else:
        assert 0
    get_fid(config, sampling_fn, score_model, eval_dir='assets/stats', job_name=args.method)

if __name__ == '__main__':
    main()