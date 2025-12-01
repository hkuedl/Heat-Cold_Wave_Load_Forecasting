import torch

import sys
print(sys.path)
sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Informer')
sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Autoformer')
import csv

import pandas as pd
from Forecasting_Models import ANN_model, LSTM_model, CNN_Model, \
    Separation_network, mmd, orthogonal_loss, Separation_network_LSTM, \
    Separation_network_CNN, NBEATS, impactnet, ConvLSTM
from Informer.informer_model import Informer
from Autoformer.autoformer_model import Autoformer
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from generate_new_samples_2D import generate_hotwave_samples, generate_coldwave_samples
import os

device = "cuda:2"
typ = torch.float
day = 19
period = 8


proportion = 1



def data_scaler(country='Belgium', strat_time = '2015/01/01/00', end_time = '2017/12/31/23'):
    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
    start_date = pd.to_datetime(strat_time)  ## Thursday
    end_date = pd.to_datetime(end_time)
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    data['Data_Hour'] = pd.to_datetime(data['Data_Hour'])  # 确保 Data_Hour 列为 datetime 类型
    data['Is_Weekend'] = data['Data_Hour'].dt.dayofweek >= 5  # 0=周一, 1=周二, ..., 6=周日
    # data['Is_Holiday'] = data['Data_Hour'].dt.date.isin(pd.to_datetime(['2015-01-01', '2015-12-25', '2016-01-01', '2016-12-25', ...]).date)  # 添加你的节假日列表
    load = np.array(data['Load'])
    temperature = np.array(data['Temperature'])

    return max(load), min(load), max(temperature), min(temperature)


def clear_diff_data(country='Belgium', strat_time = '2015/01/01/00', end_time = '2017/12/31/23'):
    # Read the data with format preparation
    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
    start_date = pd.to_datetime(strat_time)  ## Thursday
    end_date = pd.to_datetime(end_time)
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    data['Data_Hour'] = pd.to_datetime(data['Data_Hour'])  # 确保 Data_Hour 列为 datetime 类型
    data['Is_Weekend'] = data['Data_Hour'].dt.dayofweek >= 5  # 0=周一, 1=周二, ..., 6=周日
    # data['Is_Holiday'] = data['Data_Hour'].dt.date.isin(pd.to_datetime(['2015-01-01', '2015-12-25', '2016-01-01', '2016-12-25', ...]).date)  # 添加你的节假日列表

    maxload, minload, maxtem, mintem = data_scaler(country)


    load = np.array(data['Load'])
    load = (load-minload)/(maxload-minload)
    temperature = np.array(data['Temperature'])
    temperature = (temperature-mintem)/(maxtem-mintem)
    weekday = np.array(data['Is_Weekend'].astype(float))


    # define the hot wave and cold wave day
    T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                          np.min(temperature[24 * i:24 * (i + 1)])) / 2
                         for i in range(temperature.shape[0] // 24)])
    T_05 = np.percentile(T_i_list, 5)
    T_95 = np.percentile(T_i_list, 95)

    # load and temperature slices formulation
    load_slice_list = []
    tem_slice_list = []
    weekday_index_list = []
    coldwave_index = []
    hotwave_index = []
    for i in range(30, load.shape[0]//24-3-30-6):
        ## load and temperature
        load_slice_list.append(load[24 * i:24 * (i + 8)])
        tem_slice_list.append(temperature[24 * i:24 * (i + 8)])
        weekday_index_list.append(weekday[24 * (i + 7) + 1])

        ## define the cold wave index
        ECI_sig = np.mean(T_i_list[i:i + 3]) - T_05
        ECI_accl = np.mean(T_i_list[i:i + 3]) - np.mean(T_i_list[i - 30:i])
        ECF = min(0, -ECI_sig * min(-1, ECI_accl))
        coldwave_index.append(float(ECF < 0))

        ## define the hot wave index
        EHI_sig = np.mean(T_i_list[i:i + 3]) - T_95
        EHI_accl = np.mean(T_i_list[i:i + 3]) - np.mean(T_i_list[i - 30:i])
        EHF = max(0, EHI_sig * max(1, EHI_accl))
        hotwave_index.append(float(EHF > 0))

    return load_slice_list, tem_slice_list, weekday_index_list, coldwave_index, hotwave_index


def train_dataloader(country='Belgium'):
    torch.manual_seed(2)
    np.random.seed(2)
    load_slice_list, tem_slice_list, weekday_index_list, \
        coldwave_index, hotwave_index = clear_diff_data(country)

    labels = []
    for i in range(len(coldwave_index)):
        if coldwave_index[i] == 1:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [1, 0, 0])
            # print(1)
        elif hotwave_index[i] == 1:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [0, 1, 0])
        else:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [0, 0, 1])

    # print(torch.tensor(load_slice_list)[..., None].shape)
    x_data = torch.cat((torch.tensor(load_slice_list)[..., None],
                        torch.tensor(tem_slice_list)[..., None]), dim=2).type(typ)
    #print(x_data[0])
    x_data = x_data.view(x_data.shape[0], x_data.shape[1] // 24, 24, x_data.shape[2]).permute(0, 3, 1, 2)
    #print(x_data[0])
    x_data = x_data[:, :, :7, :]
    #x_data.requires_grad = True
    # y_data = torch.cat((torch.tensor(weekday_index_list)[..., None, None],
    #                    torch.tensor(coldwave_index)[..., None, None],
    #                    torch.tensor(hotwave_index)[..., None, None]), dim=2).type(typ)

    y_data = torch.tensor(labels).type(typ)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(x_train.to(device), y_train.to(device))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(x_val.to(device), y_val.to(device))
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    # print(x_data.shape)
    return train_loader, val_loader


def coldwave_dataloader(country='Belgium', coldwave_samples=None):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    #coldwave_samples = generate_coldwave_samples(country, num_samples=1200).cpu().detach().numpy()

    labels, load_slice_list, tem_slice_list = [], [], []
    ## put generated coldwave samples into the dataloader
    for i in range(coldwave_samples.shape[0]):
        labels.append(coldwave_samples[i, 0, 7:, :].flatten().tolist() +
                      coldwave_samples[i, 1, 7:, :].flatten().tolist() + [1, 0, 0])
        load_slice_list.append(coldwave_samples[i, 0, :, :].flatten())
        tem_slice_list.append(coldwave_samples[i, 1, :, :].flatten())

    ## build the dataset
    x_data = torch.cat((torch.tensor(load_slice_list)[..., None],
                        torch.tensor(tem_slice_list)[..., None]), dim=2).type(typ)
    x_data = x_data.view(x_data.shape[0], x_data.shape[1] // 24, 24, x_data.shape[2]).permute(0, 3, 1, 2)
    x_data = x_data[:, :, :7, :]
    y_data = torch.tensor(labels).type(typ)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(x_train.to(device), y_train.to(device))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(x_val.to(device), y_val.to(device))
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    # print(x_data.shape)
    return train_loader, val_loader


def train_dataloader_conditional(country='Belgium', coldwave_samples=None):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    #coldwave_samples = generate_coldwave_samples(country, num_samples=1200).cpu().detach().numpy()
    #hotwave_samples = generate_hotwave_samples(country).cpu().detach().numpy()



    load_slice_list, tem_slice_list, weekday_index_list, \
        coldwave_index, hotwave_index = clear_diff_data(country)

    labels = []

    for i in range(len(coldwave_index)):
        if coldwave_index[i] == 1:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [1, 0, 0])
            # print(1)
        elif hotwave_index[i] == 1:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [0, 1, 0])
        else:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [0, 0, 1])

    ## put generated coldwave samples into the dataloader
    for i in range(coldwave_samples.shape[0]):
        labels.append(coldwave_samples[i, 0, 7:, :].flatten().tolist() +
                      coldwave_samples[i, 1, 7:, :].flatten().tolist() + [1, 0, 0])
        load_slice_list.append(coldwave_samples[i, 0, :, :].flatten())
        tem_slice_list.append(coldwave_samples[i, 1, :, :].flatten())

    # print(torch.tensor(load_slice_list)[..., None].shape)
    x_data = torch.cat((torch.tensor(load_slice_list)[..., None],
                        torch.tensor(tem_slice_list)[..., None]), dim=2).type(typ)
    x_data = x_data.view(x_data.shape[0], x_data.shape[1] // 24, 24, x_data.shape[2]).permute(0, 3, 1, 2)
    x_data = x_data[:, :, :7, :]
    # x_data.requires_grad = True
    # y_data = torch.cat((torch.tensor(weekday_index_list)[..., None, None],
    #                    torch.tensor(coldwave_index)[..., None, None],
    #                    torch.tensor(hotwave_index)[..., None, None]), dim=2).type(typ)

    y_data = torch.tensor(labels).type(typ)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(x_train.to(device), y_train.to(device))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(x_val.to(device), y_val.to(device))
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    # print(x_data.shape)
    return train_loader, val_loader


def test_dataloader(country='Belgium'):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    load_slice_list, tem_slice_list, weekday_index_list, \
        coldwave_index, hotwave_index = clear_diff_data(country, strat_time = '2017/12/31/00', end_time = '2018/12/31/23')

    labels = []
    for i in range(len(coldwave_index)):
        if coldwave_index[i] == 1:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [1, 0, 0])
            # print(1)
        elif hotwave_index[i] == 1:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [0, 1, 0])
        else:
            labels.append(load_slice_list[i][-24:].tolist() + tem_slice_list[i][-24:].tolist() +
                          [0, 0, 1])

    # print(torch.tensor(load_slice_list)[..., None].shape)
    x_data = torch.cat((torch.tensor(load_slice_list)[..., None],
                        torch.tensor(tem_slice_list)[..., None]), dim=2).type(typ)
    x_data = x_data.view(x_data.shape[0], x_data.shape[1] // 24, 24, x_data.shape[2]).permute(0, 3, 1, 2)
    x_data = x_data[:, :, :7, :]
    #x_data.requires_grad = True
    # y_data = torch.cat((torch.tensor(weekday_index_list)[..., None, None],
    #                    torch.tensor(coldwave_index)[..., None, None],
    #                    torch.tensor(hotwave_index)[..., None, None]), dim=2).type(typ)

    y_data = torch.tensor(labels).type(typ)

    # X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    test_dataset = torch.utils.data.TensorDataset(x_data.to(device), y_data.to(device))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # print(x_data.shape)
    return test_loader





def training_process(country='Belgium', model_type='ANN', data_aug=0, coldwave_samples=None):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    if model_type == 'ANN':
        model = ANN_model(input_dim=7 * 24 * 2, output_dim=24).to(device)
    if model_type == 'LSTM':
      model = LSTM_model(input_dim=2, output_dim=24).to(device)
    if model_type == 'CNN':
      model = CNN_Model().to(device)
    #for name, param in model.named_parameters():
    #    print('values: ', param.data)

    lr = 1e-3
    if data_aug == 0:
        train_loader, val_loader = train_dataloader(country=country)
        #ratio = 'None'
    if data_aug == 1:
        train_loader, val_loader = train_dataloader_conditional(country=country, coldwave_samples=coldwave_samples)
        #ratio = proportion
    test_loader = test_dataloader(country=country)
    test_loss_list = []
    optimizer = Adam(model.parameters(), lr=lr)
    #optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer = SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    n_epochs = 2000

    # train the unconditional diffusion model
    print('model training:')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    best_val_loss = float('inf')
    count = 0

    for epoch in tqdm_epochs:
        total_train_loss = 0
        total_val_loss = 0
        num_train_samples = 0
        num_val_samples = 0

        for x, label in train_loader:
            #print(x)
            x = x.to(device)
            #print(x.shape)
            #print(label.shape)
            y = model(x, label[:, 24:48])
            #print(y)
            #loss = torch.sqrt(loss_fn(y, label[:, :24]))
            loss = loss_fn(y, label[:, :24])
            #print(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_train_samples += x.size(0)
            total_train_loss += loss.item() * x.size(0)

        #model.eval()
        for x, label in val_loader:
            with torch.no_grad():
                x = x.to(device)
                y = model(x, label[:, 24:48])
                loss = torch.sqrt(loss_fn(y, label[:, :24]))
                #print(loss)
                #loss = loss_fn(y, label[:, :24])
                num_val_samples += x.size(0)
                total_val_loss += loss.item() * x.size(0)

        forecasted_value = []
        true_value = []
        true_tem = []

        def MAE(x1, x2):
            lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
            return np.mean(lst)

        for x, label in test_loader:
            with torch.no_grad():
                x = x.to(device)
                y = model(x, label[:, 24:48])
                forecasted_value.append(y[0, :].cpu().detach().numpy())
                true_value.append(label[0, :24].cpu().detach().numpy())
                true_tem.append(label[:, 24:48].cpu().detach().numpy())

        forecasted_value = np.array(forecasted_value).flatten()[24 * day:24 * (day + period)]
        true_value = np.array(true_value).flatten()[24 * day:24 * (day + period)]
        true_tem = np.array(true_tem).flatten()[24 * day:24 * (day + period)]
        test_loss_list.append(MAE(forecasted_value, true_value))

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model_{}.pt'.format(country))
            count = 0
        else:
            count += 1
            if count >= 150:
                #print(f"Early stopping at epoch {epoch+1}")
                #print(best_val_loss)
                break


        avg_train_loss = total_train_loss / num_train_samples
        avg_val_loss = total_val_loss / num_val_samples
        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg train Loss: {avg_train_loss:5f}; Avg val Loss: {avg_val_loss:5f};"
            f"Num Samples: {num_train_samples}"
        )

        df_test_loss = pd.DataFrame(test_loss_list, columns=['My Data'])
        df_test_loss.to_excel('Convergence_curve/test_basic_{}_{}_{}.xlsx'.format(country, model_type, data_aug),
                              index=True)

    return model

def training_process_nbeats(country='Belgium', data_aug=0):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = NBEATS().to(device)
    #for name, param in model.named_parameters():
    #    print('values: ', param.data)

    lr = 1e-3
    if data_aug == 0:
        train_loader, val_loader = train_dataloader(country=country)
    if data_aug == 1:
        train_loader, val_loader = train_dataloader_conditional(country=country)
    optimizer = Adam(model.parameters(), lr=lr)
    #optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer = SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    n_epochs = 2000

    # train the unconditional diffusion model
    print('model training:')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    best_val_loss = float('inf')
    count = 0
    for epoch in tqdm_epochs:
        total_train_loss = 0
        total_val_loss = 0
        num_train_samples = 0
        num_val_samples = 0

        for x, label in train_loader:
            #print(x)
            x = x.to(device)
            #print(x.shape)
            #print(label.shape)
            y = model(x, label[:, 24:48])
            #print(y)
            #loss = torch.sqrt(loss_fn(y, label[:, :24]))
            loss = loss_fn(y, label[:, :24])
            #print(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_train_samples += x.size(0)
            total_train_loss += loss.item() * x.size(0)

        #model.eval()
        for x, label in val_loader:
            with torch.no_grad():
                x = x.to(device)
                y = model(x, label[:, 24:48])
                loss = torch.sqrt(loss_fn(y, label[:, :24]))
                #print(loss)
                #loss = loss_fn(y, label[:, :24])
                num_val_samples += x.size(0)
                total_val_loss += loss.item() * x.size(0)

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model_{}.pt'.format(country))
            count = 0
        else:
            count += 1
            if count >= 100:
                #print(f"Early stopping at epoch {epoch+1}")
                #print(best_val_loss)
                break


        avg_train_loss = total_train_loss / num_train_samples
        avg_val_loss = total_val_loss / num_val_samples
        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg train Loss: {avg_train_loss:5f}; Avg val Loss: {avg_val_loss:5f};"
            f"Num Samples: {num_train_samples}"
        )

    return model

def training_process_impactnet(country='Belgium', data_aug=0):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = impactnet().to(device)
    #for name, param in model.named_parameters():
    #    print('values: ', param.data)

    lr = 1e-3
    if data_aug == 0:
        train_loader, val_loader = train_dataloader(country=country)
    if data_aug == 1:
        train_loader, val_loader = train_dataloader_conditional(country=country)
    optimizer = Adam(model.parameters(), lr=lr)
    #optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer = SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    n_epochs = 2000

    # train the unconditional diffusion model
    print('model training:')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    best_val_loss = float('inf')
    count = 0
    for epoch in tqdm_epochs:
        total_train_loss = 0
        total_val_loss = 0
        num_train_samples = 0
        num_val_samples = 0

        for x, label in train_loader:
            #print(x)
            x = x.to(device)
            #print(x.shape)
            #print(label.shape)
            y = model(x, label[:, 24:48])
            #print(y)
            #loss = torch.sqrt(loss_fn(y, label[:, :24]))
            loss = loss_fn(y, label[:, :24])
            #print(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_train_samples += x.size(0)
            total_train_loss += loss.item() * x.size(0)

        #model.eval()
        for x, label in val_loader:
            with torch.no_grad():
                x = x.to(device)
                y = model(x, label[:, 24:48])
                loss = (loss_fn(y, label[:, :24]))
                #print(loss)
                #loss = loss_fn(y, label[:, :24])
                num_val_samples += x.size(0)
                total_val_loss += loss.item() * x.size(0)

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model_{}.pt'.format(country))
            count = 0
        else:
            count += 1
            if count >= 100:
                #print(f"Early stopping at epoch {epoch+1}")
                #print(best_val_loss)
                break


        avg_train_loss = total_train_loss / num_train_samples
        avg_val_loss = total_val_loss / num_val_samples
        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg train Loss: {avg_train_loss:5f}; Avg val Loss: {avg_val_loss:5f};"
            f"Num Samples: {num_train_samples}"
        )

    return model


def training_process_informer(country='Belgium', data_aug=0):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = Informer().to(device)
    #for name, param in model.named_parameters():
    #    print('values: ', param.data)

    lr = 1e-3
    if data_aug == 0:
        train_loader, val_loader = train_dataloader(country=country)
    if data_aug == 1:
        train_loader, val_loader = train_dataloader_conditional(country=country)
    optimizer = Adam(model.parameters(), lr=lr)
    #optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer = SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    n_epochs = 2000

    # train the unconditional diffusion model
    print('model training:')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    best_val_loss = float('inf')
    count = 0
    for epoch in tqdm_epochs:
        total_train_loss = 0
        total_val_loss = 0
        num_train_samples = 0
        num_val_samples = 0

        for x, label in train_loader:
            #print(x)
            x = x.to(device)
            #print(x.shape)
            #print(torch.unsqueeze(label[:, 24:48], dim=-1).shape)
            y = model(x, torch.unsqueeze(label[:, 24:48], dim=-1))
            #print(y)
            #loss = torch.sqrt(loss_fn(y, label[:, :24]))
            loss = loss_fn(torch.squeeze(y), label[:, :24])
            #print(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_train_samples += x.size(0)
            total_train_loss += loss.item() * x.size(0)

        #model.eval()
        for x, label in val_loader:
            with torch.no_grad():
                x = x.to(device)
                y = model(x, torch.unsqueeze(label[:, 24:48], dim=-1))
                loss = (loss_fn(torch.squeeze(y), label[:, :24]))
                #print(loss)
                #loss = loss_fn(y, label[:, :24])
                num_val_samples += x.size(0)
                total_val_loss += loss.item() * x.size(0)

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model_{}.pt'.format(country))
            count = 0
        else:
            count += 1
            if count >= 100:
                #print(f"Early stopping at epoch {epoch+1}")
                #print(best_val_loss)
                break


        avg_train_loss = total_train_loss / num_train_samples
        avg_val_loss = total_val_loss / num_val_samples
        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg train Loss: {avg_train_loss:5f}; Avg val Loss: {avg_val_loss:5f};"
            f"Num Samples: {num_train_samples}"
        )

    return model

def training_process_autoformer(country='Belgium', data_aug=0):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = Autoformer().to(device)
    #for name, param in model.named_parameters():
    #    print('values: ', param.data)

    lr = 1e-3
    if data_aug == 0:
        train_loader, val_loader = train_dataloader(country=country)
    if data_aug == 1:
        train_loader, val_loader = train_dataloader_conditional(country=country)
    optimizer = Adam(model.parameters(), lr=lr)
    #optimizer = AdamW(model.parameters(), lr=lr)
    #optimizer = SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    n_epochs = 2000

    # train the unconditional diffusion model
    print('model training:')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    best_val_loss = float('inf')
    count = 0
    for epoch in tqdm_epochs:
        total_train_loss = 0
        total_val_loss = 0
        num_train_samples = 0
        num_val_samples = 0

        for x, label in train_loader:
            #print(x)
            x = x.to(device)
            #print(x.shape)
            #print(label.shape)
            y = model(x, label[:, 24:48])
            #print(y)
            #loss = torch.sqrt(loss_fn(y, label[:, :24]))
            loss = loss_fn(y, label[:, :24])
            #print(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_train_samples += x.size(0)
            total_train_loss += loss.item() * x.size(0)

        #model.eval()
        for x, label in val_loader:
            with torch.no_grad():
                x = x.to(device)
                y = model(x, label[:, 24:48])
                loss = (loss_fn(y, label[:, :24]))
                #print(loss)
                #loss = loss_fn(y, label[:, :24])
                num_val_samples += x.size(0)
                total_val_loss += loss.item() * x.size(0)

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model_{}.pt'.format(country))
            count = 0
        else:
            count += 1
            if count >= 100:
                #print(f"Early stopping at epoch {epoch+1}")
                #print(best_val_loss)
                break


        avg_train_loss = total_train_loss / num_train_samples
        avg_val_loss = total_val_loss / num_val_samples
        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg train Loss: {avg_train_loss:5f}; Avg val Loss: {avg_val_loss:5f};"
            f"Num Samples: {num_train_samples}"
        )

    return model

def training_process_dsn(country='Belgium', model_type='ANN', coldwave_samples=None):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    if model_type == 'ANN':
        model = Separation_network(input_dim=2).to(device)
        patience = 0
        stopping_epoch = 150
        alpha = 0.5
        beta = 0.5
    if model_type == 'LSTM':
        model = Separation_network_LSTM(input_dim=2).to(device)
        patience = 0
        stopping_epoch = 150
        alpha = 1
        beta = 0.5
    if model_type == 'CNN':
        model = Separation_network_CNN(input_dim=2).to(device)
        patience = 0
        stopping_epoch = 150
        alpha = 1
        beta = 0.5

    lr = 1e-3
    train_loader1, val_loader1 = train_dataloader(country=country)
    train_loader2, val_loader2 = coldwave_dataloader(country=country, coldwave_samples=coldwave_samples)
    test_loader = test_dataloader(country=country)
    test_loss_list = []
    optimizer = Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    n_epochs = 2000

    # train the unconditional diffusion model
    print('model training:')
    tqdm_epochs = tqdm(range(1, n_epochs + 1))
    best_val_loss = float('inf')
    count = 0
    information_loss_list = []


    for epoch in tqdm_epochs:
        total_train_loss = 0
        total_val_loss = 0
        num_train_samples = 0
        num_val_samples = 0
        information_loss = 0

        for (x1, label1), (x2, label2) in zip(train_loader1, train_loader2):
            x1 = x1.to(device)
            x2 = x2.to(device)

            current_batch_size = min(x1.size(0), x2.size(0))
            x1 = x1[:current_batch_size]
            label1 = label1[:current_batch_size]
            x2 = x2[:current_batch_size]
            label2 = label2[:current_batch_size]

            # print(x1.size(0))
            # print(x2.size(0))

            hpt, hps, hct, hcs, load_t, tem_t, \
                ex_t, load_s, tem_s, ex_s, \
                prediction_s, prediction_t = model(x1, label1[:, 24:48], x2, label2[:, 24:48])

            # loss = 1*torch.sqrt(loss_fn(prediction_s, label1[:, :24]))+\
            #       1*torch.sqrt(loss_fn(prediction_t, label2[:, :24]))+\
            #       0.1*torch.sqrt(loss_fn(load_s, x1[:, 0].view(x1.shape[0], -1)))+\
            #       0.1*torch.sqrt(loss_fn(tem_s, x1[:, 1].view(x1.shape[0], -1)))+\
            #       0.1*torch.sqrt(loss_fn(ex_s, label1[:, 24:48]))+\
            #       0.1*torch.sqrt(loss_fn(load_t, x2[:, 0].view(x2.shape[0], -1)))+\
            #       0.1*torch.sqrt(loss_fn(tem_t, x2[:, 1].view(x2.shape[0], -1)))+\
            #       0.1*torch.sqrt(loss_fn(ex_t, label2[:, 24:48]))+\
            #       0.1*mmd(hct, hcs) + 0.1*orthogonal_loss(hct, hpt) + 0.1*orthogonal_loss(hcs, hps)

            loss = (
                    1 * (loss_fn(prediction_s, label1[:, :24])) +
                    1 * (loss_fn(prediction_t, label2[:, :24])) +
                    0.1 * (loss_fn(load_s, x1[:, 0].view(x1.shape[0], -1))) +
                    0.1 * (loss_fn(tem_s, x1[:, 1].view(x1.shape[0], -1))) +
                    0.1 * (loss_fn(ex_s, label1[:, 24:48])) +
                    0.1 * (loss_fn(load_t, x2[:, 0].view(x2.shape[0], -1))) +
                    0.1 * (loss_fn(tem_t, x2[:, 1].view(x2.shape[0], -1))) +
                    0.1 * (loss_fn(ex_t, label2[:, 24:48])) +
                    alpha * mmd(hct, hcs) +
                    beta * orthogonal_loss(hct, hpt) +
                    beta * orthogonal_loss(hcs, hps)
            )
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            num_train_samples += x1.size(0)
            total_train_loss += loss.item() * x1.size(0)




        # model.eval()
        for (x1, label1), (x2, label2) in zip(val_loader1, val_loader2):
            x1 = x1.to(device)
            x2 = x2.to(device)

            current_batch_size = min(x1.size(0), x2.size(0))
            x1 = x1[:current_batch_size]
            label1 = label1[:current_batch_size]
            x2 = x2[:current_batch_size]
            label2 = label2[:current_batch_size]

            with torch.no_grad():
                hpt, hps, hct, hcs, load_t, tem_t, \
                    ex_t, load_s, tem_s, ex_s, \
                    prediction_s, prediction_t = model(x1, label1[:, 24:48], x2, label2[:, 24:48])

                loss = (
                        1 * (loss_fn(prediction_s, label1[:, :24])) +
                        1 * (loss_fn(prediction_t, label2[:, :24])) +
                        0 * (loss_fn(load_s, x1[:, 0].view(x1.shape[0], -1))) +
                        0 * (loss_fn(tem_s, x1[:, 1].view(x1.shape[0], -1))) +
                        0 * (loss_fn(ex_s, label1[:, 24:48])) +
                        0 * (loss_fn(load_t, x2[:, 0].view(x2.shape[0], -1))) +
                        0 * (loss_fn(tem_t, x2[:, 1].view(x2.shape[0], -1))) +
                        0 * (loss_fn(ex_t, label2[:, 24:48])) +
                        0 * mmd(hct, hcs) +
                        0 * orthogonal_loss(hct, hpt) +
                        0 * orthogonal_loss(hcs, hps)
                )
                # print(loss)
                # loss = loss_fn(y, label[:, :24])
                num_val_samples += x1.size(0)
                total_val_loss += loss.item() * x1.size(0)

                information_loss += alpha * mmd(hct, hcs) + \
                                    beta * orthogonal_loss(hct, hpt) + \
                                    beta * orthogonal_loss(hcs, hps)


        forecasted_value = []
        true_value = []
        true_tem = []

        def MAE(x1, x2):
            lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
            return np.mean(lst)

        for x, label in test_loader:
            with torch.no_grad():
                x = x.to(device)
                h = model.shared_encoder(x, label[:, 24:48])
                h_p = model.source_encoder(x, label[:, 24:48])
                y = model.source_predictor(h, h_p)
                # y = model.source_predictor(h)
                # loss = torch.sqrt(loss_fn(y, label[:, :24]))
                # print(loss)
                # loss = loss_fn(y, label[:, :24])
                forecasted_value.append(y[0, :].cpu().detach().numpy())
                true_value.append(label[0, :24].cpu().detach().numpy())
                true_tem.append(label[:, 24:48].cpu().detach().numpy())

        forecasted_value = np.array(forecasted_value).flatten()[24 * day:24 * (day + period)]
        true_value = np.array(true_value).flatten()[24 * day:24 * (day + period)]
        true_tem = np.array(true_tem).flatten()[24 * day:24 * (day + period)]
        test_loss_list.append(MAE(forecasted_value, true_value))

        if total_val_loss < best_val_loss - patience:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), 'best_model_{}.pt'.format(country))
            count = 0
        else:
            count += 1
            if count >= stopping_epoch:
                # print(f"Early stopping at epoch {epoch+1}")
                # print(best_val_loss)
                break



        avg_train_loss = total_train_loss / num_train_samples
        avg_val_loss = total_val_loss / num_val_samples
        avg_information_loss = information_loss
        information_loss_list.append(avg_information_loss.cpu().numpy())

        tqdm_epochs.set_description(
            f"Epoch:[{epoch}]/[{n_epochs}]; Avg train Loss: {avg_train_loss:5f}; Avg val Loss: {avg_val_loss:5f};"
            f"Num Samples: {num_train_samples}"
        )

    df_information_loss = pd.DataFrame(information_loss_list, columns=['My Data'])
    df_information_loss.to_excel('Convergence_curve/convergence_{}_{}.xlsx'.format(country, model_type),
                                 index=True)
    df_test_loss = pd.DataFrame(test_loss_list, columns=['My Data'])
    df_test_loss.to_excel('Convergence_curve/test_proposed_{}_{}.xlsx'.format(country, model_type),
                                 index=True)


    return model


#training_process_dsn()










def results_visualization(country='Belgium', model_type='ANN', data_aug=0, coldwave_samples=None):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = training_process(country=country, model_type=model_type, data_aug=data_aug, coldwave_samples=coldwave_samples)
    test_loader = test_dataloader(country=country)
    loss_fn = nn.MSELoss()

    model.load_state_dict(torch.load('best_model_{}.pt'.format(country)))

    forecasted_value = []
    true_value = []
    true_tem = []

    for x, label in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = model(x, label[:, 24:48])
            loss = torch.sqrt(loss_fn(y, label[:, :24]))
            forecasted_value.append(y[0, :].cpu().detach().numpy())
            true_value.append(label[0, :24].cpu().detach().numpy())
            true_tem.append(label[:, 24:48].cpu().detach().numpy())

    forecasted_value = np.array(forecasted_value).flatten()
    true_value = np.array(true_value).flatten()
    true_tem = np.array(true_tem).flatten()

    def MAE(x1, x2):
        lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
        return np.mean(lst)

    def RMSE(x1, x2):
        lst = np.array([(x1[i] - x2[i])**2 for i in range(x1.shape[0])])
        return np.sqrt(np.mean(lst))

    df = pd.DataFrame({
        'true': true_value,
        'forecasted': forecasted_value
    })

    if data_aug == 0:
        df.to_csv('results/results_basic_{}_{}'.format(country, model_type), index=False)
    else:
        df.to_csv('results/results_basic_{}_{}_{}'.format(country, model_type, proportion), index=False)

    print('MAE: ', MAE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    print('RMSE: ', RMSE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    plt.plot(forecasted_value[24 * (day):24 * (day+60)], label='forecasted')
    plt.plot(true_value[24 * (day):24 * (day+60)], label='true')
    plt.plot(true_tem[24 * (day):24 * (day+60)], label='tem')
    plt.ylim(0, 1.2)
    plt.legend()
    plt.show()
    return MAE(forecasted_value[24 * day:24 * (day + period)],
               true_value[24 * day:24 * (day + period)]), \
        RMSE(forecasted_value[24 * day:24 * (day + period)],
             true_value[24 * day:24 * (day + period)])

def results_visualization_nbeats(country='Belgium', data_aug=0, model_type='nbeats'):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = training_process_nbeats(country=country, data_aug=data_aug)
    test_loader = test_dataloader(country=country)
    loss_fn = nn.MSELoss()

    model.load_state_dict(torch.load('best_model_{}.pt'.format(country)))

    forecasted_value = []
    true_value = []
    true_tem = []

    for x, label in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = model(x, label[:, 24:48])
            loss = torch.sqrt(loss_fn(y, label[:, :24]))
            forecasted_value.append(y[0, :].cpu().detach().numpy())
            true_value.append(label[0, :24].cpu().detach().numpy())
            true_tem.append(label[:, 24:48].cpu().detach().numpy())

    forecasted_value = np.array(forecasted_value).flatten()
    true_value = np.array(true_value).flatten()
    true_tem = np.array(true_tem).flatten()

    df = pd.DataFrame({
        'true': true_value,
        'forecasted': forecasted_value
    })

    df.to_csv('results/results_basic_{}_{}'.format(country, model_type), index=False)

    def MAE(x1, x2):
        lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
        return np.mean(lst)

    def RMSE(x1, x2):
        lst = np.array([(x1[i] - x2[i])**2 for i in range(x1.shape[0])])
        return np.sqrt(np.mean(lst))

    day = 19
    period = 8
    print('MAE: ', MAE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    print('RMSE: ', RMSE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    # plt.plot(forecasted_value[24 * (day):24 * (day+60)], label='forecasted')
    # plt.plot(true_value[24 * (day):24 * (day+60)], label='true')
    # plt.plot(true_tem[24 * (day):24 * (day+60)], label='tem')
    # plt.ylim(0, 1.2)
    # plt.legend()
    # plt.show()
    return MAE(forecasted_value[24 * day:24 * (day + period)],
               true_value[24 * day:24 * (day + period)]), \
        RMSE(forecasted_value[24 * day:24 * (day + period)],
             true_value[24 * day:24 * (day + period)])

def results_visualization_impactnet(country='Belgium', data_aug=0, model_type='impactnet'):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = training_process_impactnet(country=country, data_aug=data_aug)
    test_loader = test_dataloader(country=country)
    loss_fn = nn.MSELoss()

    model.load_state_dict(torch.load('best_model_{}.pt'.format(country)))

    forecasted_value = []
    true_value = []
    true_tem = []

    for x, label in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = model(x, label[:, 24:48])
            loss = torch.sqrt(loss_fn(y, label[:, :24]))
            forecasted_value.append(y[0, :].cpu().detach().numpy())
            true_value.append(label[0, :24].cpu().detach().numpy())
            true_tem.append(label[:, 24:48].cpu().detach().numpy())

    forecasted_value = np.array(forecasted_value).flatten()
    true_value = np.array(true_value).flatten()
    true_tem = np.array(true_tem).flatten()

    df = pd.DataFrame({
        'true': true_value,
        'forecasted': forecasted_value
    })

    df.to_csv('results/results_basic_{}_{}'.format(country, model_type), index=False)

    def MAE(x1, x2):
        lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
        return np.mean(lst)

    def RMSE(x1, x2):
        lst = np.array([(x1[i] - x2[i])**2 for i in range(x1.shape[0])])
        return np.sqrt(np.mean(lst))

    day = 19
    period = 8
    print('MAE: ', MAE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    print('RMSE: ', RMSE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    # plt.plot(forecasted_value[24 * (day):24 * (day+60)], label='forecasted')
    # plt.plot(true_value[24 * (day):24 * (day+60)], label='true')
    # plt.plot(true_tem[24 * (day):24 * (day+60)], label='tem')
    # plt.ylim(0, 1.2)
    # plt.legend()
    # plt.show()
    return MAE(forecasted_value[24 * day:24 * (day + period)],
               true_value[24 * day:24 * (day + period)]), \
        RMSE(forecasted_value[24 * day:24 * (day + period)],
             true_value[24 * day:24 * (day + period)])

def results_visualization_informer(country='Belgium', data_aug=0, model_type='informer'):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = training_process_informer(country=country, data_aug=data_aug)
    test_loader = test_dataloader(country=country)
    loss_fn = nn.MSELoss()

    model.load_state_dict(torch.load('best_model_{}.pt'.format(country)))

    forecasted_value = []
    true_value = []
    true_tem = []

    model.eval()
    for x, label in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = model(x, torch.unsqueeze(label[:, 24:48], dim=-1))
            loss = torch.sqrt(loss_fn(torch.squeeze(y), label[:, :24]))
            forecasted_value.append(y[0, :].cpu().detach().numpy())
            true_value.append(label[0, :24].cpu().detach().numpy())
            true_tem.append(label[:, 24:48].cpu().detach().numpy())

    forecasted_value = np.array(forecasted_value).flatten()
    true_value = np.array(true_value).flatten()
    true_tem = np.array(true_tem).flatten()

    def MAE(x1, x2):
        lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
        return np.mean(lst)

    def RMSE(x1, x2):
        lst = np.array([(x1[i] - x2[i])**2 for i in range(x1.shape[0])])
        return np.sqrt(np.mean(lst))

    df = pd.DataFrame({
        'true': true_value,
        'forecasted': forecasted_value
    })

    df.to_csv('results/results_basic_{}_{}'.format(country, model_type), index=False)

    day = 19
    period = 8
    print('MAE: ', MAE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    print('RMSE: ', RMSE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    # plt.plot(forecasted_value[24 * (day):24 * (day+60)], label='forecasted')
    # plt.plot(true_value[24 * (day):24 * (day+60)], label='true')
    # plt.plot(true_tem[24 * (day):24 * (day+60)], label='tem')
    # plt.ylim(0, 1.2)
    # plt.legend()
    # plt.show()
    return MAE(forecasted_value[24 * day:24 * (day + period)],
               true_value[24 * day:24 * (day + period)]), \
        RMSE(forecasted_value[24 * day:24 * (day + period)],
             true_value[24 * day:24 * (day + period)])

def results_visualization_autoformer(country='Belgium', data_aug=0, model_type='autoformer'):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = training_process_autoformer(country=country, data_aug=data_aug)
    test_loader = test_dataloader(country=country)
    loss_fn = nn.MSELoss()

    model.load_state_dict(torch.load('best_model_{}.pt'.format(country)))

    forecasted_value = []
    true_value = []
    true_tem = []

    for x, label in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = model(x, label[:, 24:48])
            loss = torch.sqrt(loss_fn(y, label[:, :24]))
            forecasted_value.append(y[0, :].cpu().detach().numpy())
            true_value.append(label[0, :24].cpu().detach().numpy())
            true_tem.append(label[:, 24:48].cpu().detach().numpy())

    forecasted_value = np.array(forecasted_value).flatten()
    true_value = np.array(true_value).flatten()
    true_tem = np.array(true_tem).flatten()

    df = pd.DataFrame({
        'true': true_value,
        'forecasted': forecasted_value
    })

    df.to_csv('results/results_basic_{}_{}'.format(country, model_type), index=False)

    def MAE(x1, x2):
        lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
        return np.mean(lst)

    def RMSE(x1, x2):
        lst = np.array([(x1[i] - x2[i])**2 for i in range(x1.shape[0])])
        return np.sqrt(np.mean(lst))

    day = 19
    period = 8
    print('MAE: ', MAE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    print('RMSE: ', RMSE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))

    # plt.plot(forecasted_value[24 * (day):24 * (day+60)], label='forecasted')
    # plt.plot(true_value[24 * (day):24 * (day+60)], label='true')
    # plt.plot(true_tem[24 * (day):24 * (day+60)], label='tem')
    # plt.ylim(0, 1.2)
    # plt.legend()
    # plt.show()
    return MAE(forecasted_value[24 * day:24 * (day + period)],
               true_value[24 * day:24 * (day + period)]), \
        RMSE(forecasted_value[24 * day:24 * (day + period)],
             true_value[24 * day:24 * (day + period)])

def results_visualization_dsn(country='Belgium', model_type='ANN', coldwave_samples=None):
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    model = training_process_dsn(country=country, model_type=model_type, coldwave_samples=coldwave_samples)
    test_loader = test_dataloader(country=country)
    loss_fn = nn.MSELoss()

    model.load_state_dict(torch.load('best_model_{}.pt'.format(country)))

    forecasted_value = []
    true_value = []
    true_tem = []

    for x, label in test_loader:
        with torch.no_grad():
            x = x.to(device)
            h = model.shared_encoder(x, label[:, 24:48])
            h_p = model.source_encoder(x, label[:, 24:48])
            y = model.source_predictor(h, h_p)
            #y = model.source_predictor(h)
            #loss = torch.sqrt(loss_fn(y, label[:, :24]))
            # print(loss)
            # loss = loss_fn(y, label[:, :24])
            forecasted_value.append(y[0, :].cpu().detach().numpy())
            true_value.append(label[0, :24].cpu().detach().numpy())
            true_tem.append(label[:, 24:48].cpu().detach().numpy())

    forecasted_value = np.array(forecasted_value).flatten()
    true_value = np.array(true_value).flatten()
    true_tem = np.array(true_tem).flatten()

    def MAE(x1, x2):
        lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
        return np.mean(lst)

    def RMSE(x1, x2):
        lst = np.array([(x1[i] - x2[i])**2 for i in range(x1.shape[0])])
        return np.sqrt(np.mean(lst))

    day = 19
    period = 8


    print('MAE: ', MAE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))
    print('RMSE: ', RMSE(forecasted_value[24 * day:24 * (day + period)], true_value[24 * day:24 * (day + period)]))

    df = pd.DataFrame({
    'true': true_value,
    'forecasted': forecasted_value
     })


    #df.to_csv('results/results_proposed_{}_{}'.format(country, model_type), index=False)
    df.to_csv('results/results_proposed_{}_{}_{}'.format(country, model_type, proportion), index=False)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    #fig, ax = plt.subplots()

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    ax.spines['top'].set_visible(False)  # 上边框
    ax.spines['right'].set_visible(False)  # 右边框

    ax.plot(forecasted_value[24 * (day):24 * (day+14)], label='forecasted')
    ax.plot(true_value[24 * (day):24 * (day+14)], label='true')

    ax.fill_between([i+24 for i in range(140)],  # x值范围
                     [0 for i in range(140)],  # 下边界（曲线本身）
                     [1.2 for i in range(140)],  # 上边界（比曲线最高点高2个单位）
                     color='blue',
                     alpha=0.1)

    ax.set_xlabel('Time index [h]')
    ax.set_ylabel('Norm. Load')
    ax.set_title('2018 Europe Coldwave')
    #plt.plot(true_tem[24 * (day):24 * (day+60)], label='tem')
    ax.set_ylim(0, 1.2)
    plt.legend(edgecolor='black')
    plt.tight_layout()
    plt.show()
    return MAE(forecasted_value[24 * day:24 * (day + period)],
               true_value[24 * day:24 * (day + period)]), \
        RMSE(forecasted_value[24 * day:24 * (day + period)],
             true_value[24 * day:24 * (day + period)])


def visualization(test_country, coldwave_samples):

    print('Results for '+str(test_country))

    resuts_list = []
    print('ANN_proposed: ')
    mae, rmse = results_visualization_dsn(country=test_country, model_type='ANN', coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('ANN: ')
    mae, rmse = results_visualization(country=test_country, model_type='ANN', data_aug=0, coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('ANN_DA: ')
    mae, rmse = results_visualization(country=test_country, model_type='ANN', data_aug=1, coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('LSTM_proposed: ')
    mae, rmse = results_visualization_dsn(country=test_country, model_type='LSTM', coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('LSTM: ')
    mae, rmse = results_visualization(country=test_country, model_type='LSTM', data_aug=0, coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('LSTM_DA: ')
    mae, rmse = results_visualization(country=test_country, model_type='LSTM', data_aug=1, coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('CNN_proposed: ')
    mae, rmse = results_visualization_dsn(country=test_country, model_type='CNN', coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('CNN: ')
    mae, rmse = results_visualization(country=test_country, model_type='CNN', data_aug=0, coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('CNN_DA: ')
    mae, rmse = results_visualization(country=test_country, model_type='CNN', data_aug=1, coldwave_samples=coldwave_samples)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('NBEATS: ')
    mae, rmse = results_visualization_nbeats(country=test_country, data_aug=0)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('ImpactNet: ')
    mae, rmse = results_visualization_impactnet(country=test_country, data_aug=0)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('Informer: ')
    mae, rmse = results_visualization_informer(country=test_country, data_aug=0)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    print('Autoformer: ')
    mae, rmse = results_visualization_autoformer(country=test_country, data_aug=0)
    resuts_list.append(mae)
    resuts_list.append(rmse)

    return resuts_list


test_country_list = ['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland']



filename = 'Europe.csv'
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(['1'])

for test_country in test_country_list:
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    coldwave_samples = generate_coldwave_samples(test_country, num_samples=int(1200*proportion), weather_type='coldwave').cpu().detach().numpy()

    results_list = visualization(test_country, coldwave_samples=coldwave_samples)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(results_list)  # 写入列表中的所有行

#visualization('Austria')