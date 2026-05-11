from tqdm import tqdm
import torch
from diff_Model_2D import SDE, VPSDE, sde_loss_fn, VESDE
from Models_2D import ScoreModel
from typing import Callable, Union, Tuple
from torch.nn import Module
from diff_Model_2D import pc_sampling, ReverseDiffusionPredictor, LangevinDynamicsCorrector, NoneCorrector
from diff_training_2D import train
from Dataset_Loader_2D import clear_extreme_data, clear_diff_data
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F


def generate_coldwave_samples(country ='Belgium', num_samples=30, weather_type = 'coldwave'):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    device = "cuda:2"
    sample_shape = (num_samples, 2, 8, 24)

    # vp_sde = VPSDE(N=1000)
    vp_sde = VESDE(N=1000)
    model = ScoreModel(vp_sde.p_0t, in_channel=2).to(device)
    country = country
    ## weather_type can be: 'hotwave', 'coldwave', 'common', or 'None'
    #weather_type = 'coldwave'

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
    #plt.ylim(0, 1.05)
    #plt.show()
    #print(samples)

    return samples


def generate_hotwave_samples(country='Belgium', num_samples=30):
    torch.manual_seed(2)
    np.random.seed(2)
    device = "cuda:2"
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
        eps=1e-5, device=device, type=weather_type
    )

    samples = samples.clamp(0., 1.)



    # plt.plot(samples.cpu().detach().numpy()[0, 0, :, :].flatten())
    # plt.plot(samples.cpu().detach().numpy()[1, 0, :, :].flatten())
    # plt.title(weather_type)
    # plt.ylim(0, 1.05)
    # plt.show()

    return samples


def generate_coldwave_only(country ='Belgium', num_samples=30, weather_type = 'coldwave'):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    device = "cuda:2"
    sample_shape = (num_samples, 2, 8, 24)

    # vp_sde = VPSDE(N=1000)
    vp_sde = VESDE(N=1000)
    model = ScoreModel(vp_sde.p_0t, in_channel=2).to(device)
    country = country
    ## weather_type can be: 'hotwave', 'coldwave', 'common', or 'None'
    #weather_type = 'coldwave'

    # train(vp_sde, model, sde_loss_fn)
    model.load_state_dict(torch.load('Model_parameters/diffusion_model_1_{}.pt'.format(country)))

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
    #plt.ylim(0, 1.05)
    #plt.show()
    #print(samples)

    return samples


def get_extreme_samples_tensor(country='Belgium', weather_type='coldwave', device='cuda:2'):
    load_slice_list, tem_slice_list, weekday_index_list, \
        coldwave_index, hotwave_index = clear_extreme_data(country, type=weather_type)

    load_arr = np.array(load_slice_list, dtype=np.float32)
    tem_arr = np.array(tem_slice_list, dtype=np.float32)

    if load_arr.ndim == 2 and load_arr.shape[1] == 192:
        load_arr = load_arr.reshape(-1, 8, 24)
    if tem_arr.ndim == 2 and tem_arr.shape[1] == 192:
        tem_arr = tem_arr.reshape(-1, 8, 24)

    samples = np.stack([load_arr, tem_arr], axis=1)  # (N, 2, 8, 24)
    return torch.tensor(samples, dtype=torch.float32).to(device)



def get_all_samples_tensor(country='Belgium', weather_type='coldwave', device='cuda:2'):
    load_slice_list, tem_slice_list, weekday_index_list, \
        coldwave_index, hotwave_index = clear_diff_data(country)

    load_arr = np.array(load_slice_list, dtype=np.float32)
    tem_arr = np.array(tem_slice_list, dtype=np.float32)

    if load_arr.ndim == 2 and load_arr.shape[1] == 192:
        load_arr = load_arr.reshape(-1, 8, 24)
    if tem_arr.ndim == 2 and tem_arr.shape[1] == 192:
        tem_arr = tem_arr.reshape(-1, 8, 24)

    samples = np.stack([load_arr, tem_arr], axis=1)  # (N, 2, 8, 24)
    return torch.tensor(samples, dtype=torch.float32).to(device)



def generate_bootstrap(country='Belgium', num_samples=30, weather_type='coldwave'):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    device = "cuda:2"

    base_samples = get_extreme_samples_tensor(country, weather_type, device)
    n_base = base_samples.shape[0]

    indices = np.random.choice(n_base, size=num_samples, replace=True)
    indices = torch.tensor(indices, dtype=torch.long, device=device)

    samples = base_samples[indices]
    samples = samples.clamp(0., 1.)
    return samples



def generate_gaussian(country='Belgium', num_samples=30, weather_type='coldwave'):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    device = "cuda:2"

    base_samples = get_extreme_samples_tensor(country, weather_type, device)
    n_base = base_samples.shape[0]

    indices = np.random.choice(n_base, size=num_samples, replace=True)
    indices = torch.tensor(indices, dtype=torch.long, device=device)

    samples = base_samples[indices].clone()

    load_std = 0.1
    temp_std = 0.1

    samples[:, 0, :, :] += torch.randn_like(samples[:, 0, :, :]) * load_std
    samples[:, 1, :, :] += torch.randn_like(samples[:, 1, :, :]) * temp_std

    samples = samples.clamp(0., 1.)
    return samples



def generate_smote(country='Belgium', num_samples=30, weather_type='coldwave'):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    device = "cuda:2"

    base_samples = get_extreme_samples_tensor(
        country=country,
        weather_type=weather_type,
        device=device
    )   # shape: (N, 2, 8, 24)

    n_base = base_samples.shape[0]
    assert n_base > 1, "Need at least two extreme samples for SMOTE interpolation."

    # flatten for nearest-neighbor search
    base_flat = base_samples.view(n_base, -1).detach().cpu().numpy()   # (N, 384)

    # kNN
    k = min(10, n_base - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(base_flat)
    distances, indices = nbrs.kneighbors(base_flat)

    synthetic_list = []

    for _ in range(num_samples):
        # 随机选一个基样本
        i = np.random.randint(0, n_base)

        # indices[i][0] 是自己，所以从 1:k+1 里选邻居
        neighbor_candidates = indices[i][1:]
        j = np.random.choice(neighbor_candidates)

        xi = base_samples[i]
        xj = base_samples[j]

        lam = np.random.uniform(0, 1.05)
        x_new = xi + lam * (xj - xi)

        synthetic_list.append(x_new.unsqueeze(0))

    samples = torch.cat(synthetic_list, dim=0)   # (num_samples, 2, 8, 24)
    #samples = samples.clamp(0., 1.)

    return samples


def generate_smote_with_noise(country='Belgium', num_samples=30, weather_type='coldwave',
                              k=10, lam_max=0.3, noise_std=0.03, smooth_kernel=5):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    device = "cuda:2"

    base_samples = get_extreme_samples_tensor(
        country=country,
        weather_type=weather_type,
        device=device
    )  # (N, 2, 8, 24)

    n_base = base_samples.shape[0]
    assert n_base > 1, "Need at least two extreme samples."

    base_flat = base_samples.view(n_base, -1).detach().cpu().numpy()

    k_eff = min(k, n_base - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(base_flat)
    distances, indices = nbrs.kneighbors(base_flat)

    synthetic_list = []

    # 1D smoothing kernel along time axis
    kernel = torch.ones(1, 1, smooth_kernel, device=device) / smooth_kernel

    for _ in range(num_samples):
        i = np.random.randint(0, n_base)
        neigh_ids = indices[i][1:]
        j = np.random.choice(neigh_ids)

        xi = base_samples[i]
        xj = base_samples[j]

        lam = np.random.uniform(0.0, lam_max)
        x_new = (1 - lam) * xi + lam * xj

        # structured noise: add noise then smooth along time axis
        noise = torch.randn_like(x_new) * noise_std

        # smooth each channel / grid separately over time dimension
        # x_new shape: (2, 8, 24) -> reshape to (1, C, T) for conv1d
        x_tmp = (x_new + noise).reshape(1, -1, 24)   # (1, 16, 24)
        x_tmp = F.pad(x_tmp, (smooth_kernel // 2, smooth_kernel // 2), mode='reflect')
        x_tmp = F.conv1d(x_tmp, kernel.repeat(x_tmp.shape[1], 1, 1), groups=x_tmp.shape[1])

        x_new = x_tmp.reshape(2, 8, 24)
        synthetic_list.append(x_new.unsqueeze(0))

    samples = torch.cat(synthetic_list, dim=0).clamp(0., 1.)
    return samples



def generate_dirichlet_mixup(country='Belgium', num_samples=30, weather_type='coldwave', k=4, alpha=0.5):
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)

    device = "cuda:2"

    base_samples = get_extreme_samples_tensor(
        country=country,
        weather_type=weather_type,
        device=device
    )  # (N, 2, 8, 24)

    n_base = base_samples.shape[0]
    assert n_base > 1, "Need at least two extreme samples."

    base_flat = base_samples.view(n_base, -1).detach().cpu().numpy()

    k_eff = min(k, n_base - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(base_flat)
    distances, indices = nbrs.kneighbors(base_flat)

    synthetic_list = []

    for _ in range(num_samples):
        i = np.random.randint(0, n_base)
        neigh_ids = indices[i][1:]   # exclude itself

        # choose m samples including the anchor itself
        m = min(k_eff + 1, len(neigh_ids) + 1)
        chosen = [i] + list(np.random.choice(neigh_ids, size=m - 1, replace=False))

        # Dirichlet weights
        w = np.random.dirichlet([alpha] * m)

        x_new = torch.zeros_like(base_samples[0])
        for weight, idx in zip(w, chosen):
            x_new += float(weight) * base_samples[idx]

        synthetic_list.append(x_new.unsqueeze(0))

    samples = torch.cat(synthetic_list, dim=0).clamp(0., 1.)
    return samples

#generate_coldwave_samples(country='Belgium')
#plt.plot(samples.cpu()[0, 0, :])