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

    def replace_hour(date_hour_str):
        parts = date_hour_str.rsplit('/', 1)  # 从右侧分割，最多分割一次
        hour = int(parts[-1])  # 获取小时部分并转换为整数

        # 如果小时在 1 到 24 之间，递减小时
        if 1 <= hour <= 24:
            new_hour = (hour - 1) if hour != 1 else 0  # 如果小时为 1，则替换为 0
            return parts[0] + f'/{new_hour:02d}'  # 格式化为两位数
        if hour == 25:
            new_hour = 23
            return parts[0] + f'/{new_hour:02d}'
        return date_hour_str  # 如果没有找到有效的小时，返回原字符串

    data['Date_Hour'] = data['Date_Hour'].apply(replace_hour)

    start_date = pd.to_datetime(strat_time)  ## Thursday
    end_date = pd.to_datetime(end_time)
    data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]

    data['Date_Hour'] = pd.to_datetime(data['Date_Hour'])  # 确保 Data_Hour 列为 datetime 类型
    #data['Is_Weekend'] = data['Data_Hour'].dt.dayofweek >= 5  # 0=周一, 1=周二, ..., 6=周日
    # data['Is_Holiday'] = data['Data_Hour'].dt.date.isin(pd.to_datetime(['2015-01-01', '2015-12-25', '2016-01-01', '2016-12-25', ...]).date)  # 添加你的节假日列表
    load = np.array(data['Load'])
    temperature = np.array(data['Temperature'])

    # 过滤掉load中的零值
    load_nonzero = load[load != 0]

    # 如果过滤后数组不为空，使用非零值的最小值；否则使用0或其他默认值
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
        # 第二个日期没有时间，默认为00:00
        datetime2 = datetime.strptime(date2, '%Y/%m/%d')
        days_difference = abs((datetime2 - datetime1).days)

    def replace_hour(date_hour_str):
        parts = date_hour_str.rsplit('/', 1)  # 从右侧分割，最多分割一次
        hour = int(parts[-1])  # 获取小时部分并转换为整数

        # 如果小时在 1 到 24 之间，递减小时
        if 1 <= hour <= 24:
            new_hour = (hour - 1) if hour != 1 else 0  # 如果小时为 1，则替换为 0
            return parts[0] + f'/{new_hour:02d}'  # 格式化为两位数
        if hour == 25:
            new_hour = 23
            return parts[0] + f'/{new_hour:02d}'
        return date_hour_str  # 如果没有找到有效的小时，返回原字符串

    data['Date_Hour'] = data['Date_Hour'].apply(replace_hour)

    start_date = pd.to_datetime(strat_time)  ## Thursday
    end_date = pd.to_datetime(end_time)
    data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]


    data['Date_Hour'] = pd.to_datetime(data['Date_Hour'])  # 确保 Data_Hour 列为 datetime 类型
    #data['Is_Weekend'] = data['Data_Hour'].dt.dayofweek >= 5  # 0=周一, 1=周二, ..., 6=周日
    # data['Is_Holiday'] = data['Data_Hour'].dt.date.isin(pd.to_datetime(['2015-01-01', '2015-12-25', '2016-01-01', '2016-12-25', ...]).date)  # 添加你的节假日列表

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