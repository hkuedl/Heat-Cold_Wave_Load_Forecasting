import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import csv
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split


device = "cuda:3"
typ = torch.float32

def data_scaler(country='Belgium', strat_time = '2021/03/28/00', end_time = '2023/05/31/23'):
    data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(country), header=0, usecols=['Date_Hour', 'Load', 'Temperature'])

    ## Handling daylight saving time and standard time
    def replace_hour(date_hour_str):
        parts = date_hour_str.rsplit('/', 1) 
        hour = int(parts[-1])  # Extract the hour portion and convert it to an integer

        # If the hour is between 1 and 24, decrease the hour.
        if 1 <= hour <= 24:
            new_hour = (hour - 1) if hour != 1 else 0  
            return parts[0] + f'/{new_hour:02d}'  
        if hour == 25:
            new_hour = 23
            return parts[0] + f'/{new_hour:02d}'
        return date_hour_str  

    data['Date_Hour'] = data['Date_Hour'].apply(replace_hour)

    start_date = pd.to_datetime(strat_time)  ## Thursday
    end_date = pd.to_datetime(end_time)
    data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]

    data['Date_Hour'] = pd.to_datetime(data['Date_Hour'])  # ensure datetime type
    load = np.array(data['Load'])
    temperature = np.array(data['Temperature'])

    # filter missing values
    load_nonzero = load[load != 0]
    load_min = load_nonzero.min() if len(load_nonzero) > 0 else 0

    return max(load), load_min, max(temperature), min(temperature)




def clear_diff_data(country='Belgium', strat_time = '2021/03/28/00', end_time = '2023/05/31/23'):
    # Read the data with format preparation
    data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(country), header=0,
                       usecols=['Date_Hour', 'Load', 'Temperature'])

    from datetime import datetime

    if strat_time > '2023/07/01':
        days_difference = 10000
    else:
        date1 = strat_time
        date2 = '2023/07/01'
        datetime1 = datetime.strptime(date1, '%Y/%m/%d/%H')
        datetime2 = datetime.strptime(date2, '%Y/%m/%d')
        days_difference = abs((datetime2 - datetime1).days)

    def replace_hour(date_hour_str):
        parts = date_hour_str.rsplit('/', 1) 
        hour = int(parts[-1])  

        if 1 <= hour <= 24:
            new_hour = (hour - 1) if hour != 1 else 0  
            return parts[0] + f'/{new_hour:02d}'  
        if hour == 25:
            new_hour = 23
            return parts[0] + f'/{new_hour:02d}'
        return date_hour_str  # If no valid hour is found, return the original string.

    data['Date_Hour'] = data['Date_Hour'].apply(replace_hour)

    start_date = pd.to_datetime(strat_time)  ## Thursday
    end_date = pd.to_datetime(end_time)
    data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]


    data['Date_Hour'] = pd.to_datetime(data['Date_Hour']) 

    maxload, minload, maxtem, mintem = data_scaler(country)


    load = np.array(data['Load'])
    #load = (load-minload)/(maxload-minload)/1.15
    temperature = np.array(data['Temperature'])
    temperature = (temperature-mintem)/(maxtem-mintem)
    #weekday = np.array(data['Is_Weekend'].astype(float))


    # define the hot wave and cold wave day
    T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                          np.min(temperature[24 * i:24 * (i + 1)])) / 2
                         for i in range(temperature.shape[0] // 24)])
    T_05 = np.percentile(T_i_list, 5)
    T_95 = np.percentile(T_i_list, 95)
    #print(T_95)

    # load and temperature slices formulation
    load_slice_list = []
    tem_slice_list = []
    weekday_index_list = []
    coldwave_index = []
    hotwave_index = []
    for i in range(30, load.shape[0]//24-3-30-6):
        ## load and temperature
        #if load[24 * i:24 * (i + 8)].min() == 0:
        #    continue

        #if days_difference+38 >= i >= days_difference:
        #    continue
        load_slice_list.append((load[24 * i:24 * (i + 8)]-minload)/(maxload-minload))
        tem_slice_list.append(temperature[24 * i:24 * (i + 8)])
        #weekday_index_list.append(weekday[24 * (i + 7) + 1])

        ## define the cold wave index
        ECI_sig = np.mean(T_i_list[i+7:i + 8]) - T_05
        ECI_accl = np.mean(T_i_list[i+7:i + 10]) - np.mean(T_i_list[i - 23:i+7])
        ECF = min(0, -ECI_sig * min(-1, ECI_accl))
        coldwave_index.append(float(ECF < 0))

        ## define the hot wave index
        EHI_sig = np.mean(T_i_list[i+7:i + 8]) - T_95
        EHI_accl = np.mean(T_i_list[i+7:i + 10]) - np.mean(T_i_list[i - 23:i+7])
        EHF = max(0, EHI_sig * max(1, EHI_accl))
        hotwave_index.append(float(EHF > 0))

    return load_slice_list, tem_slice_list, weekday_index_list, coldwave_index, hotwave_index



def diff_dataloader(country='Belgium'):
    load_slice_list, tem_slice_list, weekday_index_list, \
        coldwave_index, hotwave_index = clear_diff_data(country)

    labels = []
    for i in range(len(coldwave_index)):
        if coldwave_index[i] == 1:
            labels.append([1, 0, 0])
            #print(1)
        elif hotwave_index[i] == 1:
            labels.append([0, 1, 0])
        else:
            labels.append([0, 0, 1])


    #print(torch.tensor(load_slice_list)[..., None].shape)
    x_data = torch.cat((torch.tensor(load_slice_list)[..., None],
                        torch.tensor(tem_slice_list)[..., None]), dim=2).type(typ)
    x_data = x_data.view(x_data.shape[0], x_data.shape[1]//24, 24, x_data.shape[2]).permute(0, 3, 1, 2)
    x_data.requires_grad = True
    #y_data = torch.cat((torch.tensor(weekday_index_list)[..., None, None],
    #                    torch.tensor(coldwave_index)[..., None, None],
    #                    torch.tensor(hotwave_index)[..., None, None]), dim=2).type(typ)

    #print(labels)
    y_data = torch.tensor(labels).type(typ)

    #X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    train_dataset = torch.utils.data.TensorDataset(x_data.to(device), y_data.to(device))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #print(x_data.shape)
    return train_loader


clear_diff_data('Allegheny Power System')

#diff_dataloader(country='Allegheny Power System')
