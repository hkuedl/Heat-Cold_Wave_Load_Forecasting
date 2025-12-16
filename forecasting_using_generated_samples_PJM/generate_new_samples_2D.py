from tqdm import tqdm
import torch
from diff_Model_2D import SDE, VPSDE, sde_loss_fn, VESDE
from Models_2D import ScoreModel
from typing import Callable, Union, Tuple
from torch.nn import Module
from diff_Model_2D import pc_sampling, ReverseDiffusionPredictor, LangevinDynamicsCorrector, NoneCorrector
from diff_training_2D import train
import matplotlib.pyplot as plt
import numpy as np
import random


def generate_coldwave_samples(country ='Belgium', num_samples=100):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    device = "cuda:3"
    sample_shape = (num_samples, 2, 8, 24)

    # vp_sde = VPSDE(N=1000)
    vp_sde = VESDE(N=1000)
    model = ScoreModel(vp_sde.p_0t, in_channel=2).to(device)
    country = country
    ## weather_type can be: 'hotwave', 'coldwave', 'common', or 'None'
    weather_type = 'hotwave'

    # train(vp_sde, model, sde_loss_fn)
    model.load_state_dict(torch.load('Model_parameters/diffusion_model_{}.pt'.format(country)))

    predictor = ReverseDiffusionPredictor(vp_sde, model, weather_type)
    corrector = LangevinDynamicsCorrector(
        vp_sde, model,
        snr=0.2, n_steps=1
    )
    # corrector = NoneCorrector(vp_sde, model, snr=0.1, n_steps=1)

    samples = pc_sampling(
        vp_sde, sample_shape,
        predictor.update_fn, corrector.update_fn,
        eps=2e-5, device=device, type=weather_type
    )

    samples = samples.clamp(0., 1.)

    #plt.plot(samples.cpu().detach().numpy()[0, 0, :, :].flatten())
    #for i in range(10):
    #    plt.plot(samples.cpu().detach().numpy()[i, 0, :, :].flatten())
    #plt.plot(samples.cpu().detach().numpy()[1, 0, :, :].flatten())
    #plt.title(weather_type)
    #plt.ylim(0, 1.2)
    #plt.show()
    #print(samples)

    return samples


def generate_hotwave_samples(country='Belgium', num_samples=50):

    device = "cuda:3"
    sample_shape = (num_samples, 2, 8, 24)

    # vp_sde = VPSDE(N=1000)
    vp_sde = VESDE(N=1000)
    model = ScoreModel(vp_sde.p_0t, in_channel=2).to(device)
    country = country
    ## weather_type can be: 'hotwave', 'coldwave', 'common', or 'None'
    weather_type = 'common'

    # train(vp_sde, model, sde_loss_fn)
    model.load_state_dict(torch.load('Model_parameters/diffusion_model_{}.pt'.format(country)))

    predictor = ReverseDiffusionPredictor(vp_sde, model, weather_type)
    corrector = LangevinDynamicsCorrector(
        vp_sde, model,
        snr=0.2, n_steps=1
    )
    # corrector = NoneCorrector(vp_sde, model, snr=0.1, n_steps=1)

    samples = pc_sampling(
        vp_sde, sample_shape,
        predictor.update_fn, corrector.update_fn,
        eps=2e-5, device=device, type=weather_type
    )

    samples = samples.clamp(0., 1.)

    #plt.plot(samples.cpu().detach().numpy()[0, 0, :, :].flatten())
    #for i in range(20):
    #   plt.plot(samples.cpu().detach().numpy()[i, 0, :, :].flatten())
    #plt.plot(samples.cpu().detach().numpy()[1, 0, :, :].flatten())
    #plt.title(weather_type)
    #plt.ylim(0, 1.2)
    #plt.show()

    return samples


#generate_hotwave_samples(country='Allegheny Power System', num_samples=50)

