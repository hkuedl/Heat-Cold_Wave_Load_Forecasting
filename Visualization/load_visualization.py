import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import csv

country='France'

data=pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
start_date = pd.to_datetime('2015/01/01/00')
end_date = pd.to_datetime('2019/4/30/23')
data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

load = np.array(data['Load'])
temperature = np.array(data['Temperature'])

T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)])+
                     np.min(temperature[24 * i:24 * (i + 1)]))/2
                     for i in range(temperature.shape[0] // 24)])
T_05 = np.percentile(T_i_list, 5)
print(T_05)
ECI_sig = np.array([np.mean(T_i_list[i+30:i+30+3])-T_05
                    for i in range(T_i_list.shape[0]-3-30)])
ECI_accl = np.array([np.mean(T_i_list[i+30:i+30+3])-np.mean(T_i_list[i:i+30])
                     for i in range(T_i_list.shape[0]-3-30)])
ECF = np.array([min(0, -ECI_sig[i]*min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

coldwave_day_index = []
for i in range(ECF.shape[0]):
    if ECF[i] < 0:
        coldwave_day_index.append(i+30)

print(coldwave_day_index)
print(len(coldwave_day_index))


def average_relationship():
    average_day_load = [np.max(load[24 * i:24 * (i + 1)]) for i in range(load.shape[0] // 24 - 1)]
    average_day_temperature = [np.mean(temperature[24 * i:24 * (i + 1)]) for i in range(temperature.shape[0] // 24 - 1)]

    plt.scatter(average_day_temperature, average_day_load)
    plt.title('Average_tem-Average_load')
    plt.show()

def max_relationship():
    max_day_load = [np.max(load[24 * i:24 * (i + 1)]) for i in range(load.shape[0] // 24 - 1)]
    max_day_temperature = [np.max(temperature[24 * i:24 * (i + 1)]) for i in range(temperature.shape[0] // 24 - 1)]

    plt.scatter(max_day_temperature, max_day_load)
    plt.title('Max_tem-Max_load')
    plt.show()

def min_relationship():
    max_day_load = [np.max(load[24 * i:24 * (i + 1)]) for i in range(load.shape[0] // 24 - 1)]
    max_day_temperature = [np.min(temperature[24 * i:24 * (i + 1)]) for i in range(temperature.shape[0] // 24 - 1)]

    plt.scatter(max_day_temperature, max_day_load)
    plt.title('Min_tem-Max_load')
    plt.show()

average_relationship()
max_relationship()
min_relationship()

def plot_week_relationship():
    day = coldwave_day_index[36]
    print(day)
    #day = 180
    fig, axes = plt.subplots(1, 1, figsize=(6, 3), layout="constrained")
    ax1 = axes
    ax11 = ax1.twinx()
    ax1.plot(load[day*24:day*24+ 24*7*3], label='load', color='crimson')
    ax11.plot(temperature[day*24:day*24+ 24*7*3], label='temperature', color='cadetblue')
    plt.legend()
    plt.show()

def vis_unregular():
    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
    start_date = pd.to_datetime('2021/12/01/00')
    end_date = pd.to_datetime('2021/12/07/23')
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    load = np.array(data['Load'])
    load = (load-min(load))/(max(load)-min(load))/1.15
    temperature = np.array(data['Temperature'])
    temperature = (temperature-min(temperature))/(max(temperature)-min(temperature))/1.1
    #weekday = np.array(data['Is_Weekend'].astype(float))

#plot_week_relationship()



