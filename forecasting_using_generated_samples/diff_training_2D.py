from tqdm import tqdm
import torch
from Dataset_Loader_2D import diff_dataloader
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from diff_Model_2D import SDE, VPSDE, sde_loss_fn, VESDE, classifier_fn
from typing import Callable, Union, Tuple
from torch.nn import Module
from Models_2D import ScoreModel, ClassifierModel
import numpy as np
import random
import os

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
device = "cuda:2"

lr = 1e-3
n_epochs = 1000


def train(sde: SDE, model: Module, loss_fn: Callable, country='Belgium'):
    data_loader = diff_dataloader(country=country)
    optimizer = Adam(model.parameters(), lr=lr)
    #optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer = SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # train the unconditional diffusion model
    print('Diffusion model training:')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    best_loss = 10000

    for epoch in tqdm_epochs:
        total_loss = 0.
        num_samples = 0

        for x, y in data_loader:
            x = x.to(device)
            loss = loss_fn(sde, model, x, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_samples += x.size(0)
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / num_samples
        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg Loss: {avg_loss:5f}; "
            f"Num Samples: {num_samples}"
        )

        if avg_loss <= best_loss:
            torch.save(model.state_dict(), 'Model_parameters/diffusion_model_{}.pt'.format(country))
            best_loss = avg_loss

    #torch.save(model.state_dict(), 'Model_parameters/diffusion_model_{}.pt'.format(country))



def classifier_training(sde: SDE, model: Module,
                        classifier: Module, loss_fn: Callable,
                        classifier_fn: Callable, country='Belgium'):
    data_loader = diff_dataloader(country=country)
    optimizer = Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # train the Classifier model
    print('Classifier training: ')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    for epoch in tqdm_epochs:
        total_loss = 0.
        num_samples = 0

        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            loss = classifier_fn(sde, classifier, x, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_samples += x.size(0)
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / num_samples
        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg Loss: {avg_loss:5f}; "
            f"Num Samples: {num_samples}"
        )

    optimizer = Adam(classifier.parameters(), lr=1e-4)
    optimizer.zero_grad()

test_country_list = ['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland']

#test_country_list = ['France']
#for country in test_country_list:

#    vp_sde = VESDE(N=1000)
#    model = ScoreModel(vp_sde.p_0t, in_channel=2).to(device)
#    model = train(vp_sde, model, sde_loss_fn, country=country)


