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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch, Rectangle


device = "cuda:2"
typ = torch.float

test_country_list = ['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland']


def coldwave_scaler(country='Belgium', strat_time = '2015/01/01/00', end_time = '2017/12/31/23'):
    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))


    start_date = pd.to_datetime(strat_time)  ## Thursday
    end_date = pd.to_datetime(end_time)
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    data['Data_Hour'] = pd.to_datetime(data['Data_Hour'])  # 确保 Data_Hour 列为 datetime 类型
    #data['Is_Weekend'] = data['Data_Hour'].dt.dayofweek >= 5  # 0=周一, 1=周二, ..., 6=周日
    # data['Is_Holiday'] = data['Data_Hour'].dt.date.isin(pd.to_datetime(['2015-01-01', '2015-12-25', '2016-01-01', '2016-12-25', ...]).date)  # 添加你的节假日列表
    load = np.array(data['Load'])
    temperature = np.array(data['Temperature'])

    return max(load), min(load), max(temperature), min(temperature)



def plot_curve(titlesize=16, ticksize=14, labelsize=14):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    country = 'France'
    model_type = 'LSTM'
    start_p = 456
    period = 24 * 7

    text_font = {'size': 15}
    num_font = {'size': 12}




    df = pd.read_csv('results/results_basic_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    true = df.values[:, 0]
    baseline_forecasted = df.values[:, 1]
    df = pd.read_csv('results/results_proposed_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    proposed_forecasted = df.values[:, 1]

    print(df.head())

    fig, ax = plt.subplots(figsize=(8, 4.5))

    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))

    start_date = pd.to_datetime('2017/09/01/00')
    end_date = pd.to_datetime('2018/08/31/23')
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    maxload, minload, maxtem, mintem = coldwave_scaler(country)

    load = np.array(data['Load'])
    load = (load - minload) / (maxload - minload)
    temperature = np.array(data['Temperature'])
    temperature = (temperature - mintem) / (maxtem - mintem)

    tem = [(np.max(temperature[24 * i:24 * (i + 1)]) +
            np.min(temperature[24 * i:24 * (i + 1)])) / 2 for i in range(temperature.shape[0] // 24)]


    x1 = np.linspace(0, load.shape[0], load.shape[0])
    line1, = ax.plot(x1, load, label='Load', color='#88A0DCFF', lw=0.3)
    ax.set_ylim([0, 1.2])
    #ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Norm. Load', fontsize=15)




    # 添加小图（inset_axes）
    ax_inset = inset_axes(
    ax,
    width="35%", height="40%",
    loc="center left",          # 小图的锚点位置
    bbox_to_anchor=(0.8, 0.6, 1, 1), # 锚点偏移（x=1.05 表示主图右侧外）
    bbox_transform=ax.transAxes, # 使用主图坐标系
    borderpad=0
    )

    #ax_inset = ax.inset_axes((0.8, 0.1, 0.35, 0.4))



    ax_inset.plot(true[start_p:start_p + period], color="blue", label="Ground Truth", lw=2)
    ax_inset.plot(baseline_forecasted[start_p:start_p + period], color="crimson", label="Baseline", ls='--')
    ax_inset.plot(proposed_forecasted[start_p:start_p + period], color="#46AEA0FF", label="Proposed", ls='--')
    ax_inset.set_title("Coldwave Period", fontsize=15)

    tx0 = 0
    tx1 = 24*7
    ty0 = 0.45
    ty1 = 1.08

    ax_inset.set_xlim(tx0, tx1)
    ax_inset.set_ylim(ty0, ty1)

    sx = [4268, period+4268, period+4268, 4268, 4268]
    sy = [0.6, 0.6, 1, 1, 0.6]

    ax.plot(sx, sy, "black")
    xy = (4268+period, 1)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax_inset, axesB=ax, ls='--', color='gray')
    #con.set_color('silver')
    ax_inset.add_artist(con)


    xy = (4268 + period, 0.6)
    xy2 = (tx0, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax_inset, axesB=ax, ls='--', color='gray')
    #con.set_color('silver')
    ax_inset.add_artist(con)






    ax_inset_2 = inset_axes(
        ax,
        width="35%", height="40%",
        loc="center left",  # 小图的锚点位置
        bbox_to_anchor=(0.8, -0.1, 1, 1),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax.fill_between([4100 + i for i in range(300)],  # x值范围
                     [0 for i in range(300)],  # 下边界（曲线本身）
                     [1.05 for i in range(300)],  # 上边界（比曲线最高点高2个单位）
                     color='blue',
                     alpha=0.1)

    ax.annotate('2018 Coldwave',
                 xy=(4100, 0.2),  # 箭头终点位置
                 xytext=(3400, 0.2),  # 箭头起点位置（向右偏移300单位）
                 arrowprops=dict(arrowstyle='->', color='black', lw=2),
                 ha='right', fontsize=15)

    ax_inset_2.plot(true[start_p+8*24:start_p + period+8*24], color="blue", label="Ground Truth", lw=2)
    ax_inset_2.plot(baseline_forecasted[start_p+8*24:start_p + period+8*24], color="crimson", label="Baseline", ls='--')
    ax_inset_2.plot(proposed_forecasted[start_p+8*24:start_p + period+8*24], color="#46AEA0FF", label="Proposed", ls='--')
    ax_inset_2.set_title("Common Period", fontsize=15)

    tx0 = 0
    tx1 = 24 * 7
    ty0 = 0.3
    ty1 = 0.7

    ax_inset_2.set_xlim(tx0, tx1)
    ax_inset_2.set_ylim(ty0, ty1)

    sx = [4268+period, period*2 + 4268, period*2 + 4268, 4268+period, 4268+period]
    sy = [0.3, 0.3, 0.7, 0.7, 0.3]

    ax.plot(sx, sy, "black")
    xy = (4268 + period*2, 0.7)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax_inset_2, axesB=ax, ls='--', color='gray')
    # con.set_color('silver')
    ax_inset_2.add_artist(con)

    xy = (4268 + period, 0.3)
    xy2 = (tx0, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=ax_inset_2, axesB=ax, ls='--', color='gray')
    # con.set_color('silver')
    ax_inset_2.add_artist(con)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax_inset.spines['top'].set_visible(False)
    #ax_inset.spines['right'].set_visible(False)
    #ax_inset_2.spines['top'].set_visible(False)
    #ax_inset_2.spines['right'].set_visible(False)

    # 0是09/01
    x = [0+30*24, 0+(30+31+30+31)*24, 0+(30+31+30+31+28+31+30)*24, load.shape[0]-1-31*24-31*24]
    labels = ['2017 Oct.', '2018 Jan.', '2018 Apr.', '2018 Jul.']
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')  # 旋转45度

    ax.tick_params(axis='both', labelsize=15)  # 同时设置x轴和y轴刻度标签
    ax_inset.tick_params(axis='both', labelsize=15)  # 同时设置x轴和y轴刻度标签
    ax_inset_2.tick_params(axis='both', labelsize=15)  # 同时设置x轴和y轴刻度标签


    # 方法1：自动调整布局（推荐）
    plt.tight_layout(rect=[0, 0.1, 0.85, 0.8])  # 右侧留 20% 空白
    handles_inset, labels_inset = ax_inset.get_legend_handles_labels()
    ax.legend(handles_inset, labels_inset,
              ncols=3, fontsize=15,
              bbox_to_anchor=(0.5, 0.95),
              frameon=True, edgecolor='white')
    #ax.set_title("Europe - France", fontsize=15, pad=55)
    fig.suptitle("Forecasting Results with/without Generated Samples",
                 y=1, fontsize=titlesize)


    plt.show()

def plot_curve_simple():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    country = 'France'
    model_type = 'LSTM'
    start_p = 456
    period = 24 * 7

    text_font = {'size': 15}
    num_font = {'size': 12}




    df = pd.read_csv('results/results_basic_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    true = df.values[:, 0]
    baseline_forecasted = df.values[:, 1]
    df = pd.read_csv('results/results_proposed_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    proposed_forecasted = df.values[:, 1]

    print(df.head())

    fig, ax = plt.subplots(figsize=(8, 4.5))

    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))

    start_date = pd.to_datetime('2017/09/01/00')
    end_date = pd.to_datetime('2018/08/31/23')
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    maxload, minload, maxtem, mintem = coldwave_scaler(country)

    load = np.array(data['Load'])
    load = (load - minload) / (maxload - minload)
    temperature = np.array(data['Temperature'])
    temperature = (temperature - mintem) / (maxtem - mintem)

    tem = [(np.max(temperature[24 * i:24 * (i + 1)]) +
            np.min(temperature[24 * i:24 * (i + 1)])) / 2 for i in range(temperature.shape[0] // 24)]


    x1 = np.linspace(0, load.shape[0], load.shape[0])
    line1, = ax.plot(x1, load, label='Load', color='#88A0DCFF', lw=0.3)
    ax.set_ylim([0, 1.2])
    #ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Norm. Load', fontsize=15)



    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax_inset.spines['top'].set_visible(False)
    #ax_inset.spines['right'].set_visible(False)
    #ax_inset_2.spines['top'].set_visible(False)
    #ax_inset_2.spines['right'].set_visible(False)

    # 0是09/01
    x = [0 + 30 * 24, 0 + (30 + 31 + 30 + 31) * 24, 0 + (30 + 31 + 30 + 31 + 28 + 31 + 30) * 24,
         load.shape[0] - 1 - 31 * 24 - 31 * 24]
    labels = ['2017 Oct.', '2018 Jan.', '2018 Apr.', '2018 Jul.']
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')  # 旋转45度

    ax.tick_params(axis='both', labelsize=15)  # 同时设置x轴和y轴刻度标签


    # 方法1：自动调整布局（推荐）
    #plt.tight_layout(rect=[0, 0.1, 0.85, 0.8])  # 右侧留 20% 空白
    ax.set_title("Europe - France", fontsize=15, pad=55)


    plt.show()



#plot_curve()

#plot_curve_simple()


def plot_before_coldwave():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    country = 'Norway'
    model_type = 'LSTM'
    start_p = 456
    period = 24 * 7

    text_font = {'size': 15}
    num_font = {'size': 12}

    df = pd.read_csv('results/results_basic_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    true = df.values[:, 0]
    baseline_forecasted = df.values[:, 1]
    df = pd.read_csv('results/results_proposed_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    proposed_forecasted = df.values[:, 1]

    print(df.head())

    fig, ax = plt.subplots(2, 1, figsize=(4.5, 3.5), sharex=True)
    ax1 = ax[0]
    ax2 = ax[1]

    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))

    start_date = pd.to_datetime('2017/09/01/00')
    end_date = pd.to_datetime('2018/08/31/23')
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    maxload, minload, maxtem, mintem = coldwave_scaler(country)

    load = np.array(data['Load'])
    load = (load - minload) / (maxload - minload)
    temperature = np.array(data['Temperature'])
    temperature = (temperature - mintem) / (maxtem - mintem)

    tem = [(np.max(temperature[24 * i:24 * (i + 1)]) +
            np.min(temperature[24 * i:24 * (i + 1)])) / 2 for i in range(temperature.shape[0] // 24)]


    line1, = ax1.plot(true[start_p:start_p+period], label='Ground Truth', color='black', lw=1, ls='--')
    line2, = ax1.plot(baseline_forecasted[start_p:start_p+period], label='Baseline', color='#FF162A', lw=1.2)
    line3, = ax1.plot(proposed_forecasted[start_p:start_p+period], label='Proposed', color='#035AA6FF', lw=0.9)
    ax1.set_ylabel('Load')
    ax1.set_title('Coldwave Forecasting Results')

    ax2.plot(np.abs(true[start_p:start_p+period]-baseline_forecasted[start_p:start_p+period]),
             color='#FF162A', lw=1, alpha=1)
    ax2.plot(np.abs(true[start_p:start_p + period] - proposed_forecasted[start_p:start_p + period]),
             color='#035AA6FF',
             lw=1, alpha=1)
    ax2.set_ylabel('Deviation')
    ax2.set_yticks([0, 0.1, 0.2])
    ax2.set_ylim(0, 0.2)

    plt.tight_layout()
    plt.show()


#plot_before_coldwave()

def plot_day_by_day(titlesize=16, ticksize=16, labelsize=16):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    country = 'Croatia'
    model_type = 'LSTM'
    start_p = 456+24*7
    period = 24*1

    text_font = {'size': 15}
    num_font = {'size': 12}

    df = pd.read_csv('results/results_basic_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    true = df.values[:, 0]
    baseline_forecasted = df.values[:, 1]
    df = pd.read_csv('results/results_proposed_{}_{}'.format(country, model_type), encoding='utf-8', header=0)
    proposed_forecasted = df.values[:, 1]

    print(df.head())

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey='row')
    ax1 = ax[0][0]
    ax2 = ax[1][0]
    ax3 = ax[0][1]
    ax4 = ax[1][1]
    ax5 = ax[0][2]
    ax6 = ax[1][2]

    ax1.grid(axis='y',  # 仅y轴方向（水平线）
            linestyle='--',
            alpha=0.7,
            color='gray')
    ax2.grid(axis='y',  # 仅y轴方向（水平线）
             linestyle='--',
             alpha=0.7,
             color='gray')
    ax3.grid(axis='y',  # 仅y轴方向（水平线）
             linestyle='--',
             alpha=0.7,
             color='gray')
    ax4.grid(axis='y',  # 仅y轴方向（水平线）
             linestyle='--',
             alpha=0.7,
             color='gray')
    ax5.grid(axis='y',  # 仅y轴方向（水平线）
             linestyle='--',
             alpha=0.7,
             color='gray')
    ax6.grid(axis='y',  # 仅y轴方向（水平线）
             linestyle='--',
             alpha=0.7,
             color='gray')

    line1, = ax1.plot(true[start_p:start_p + period], label='Ground Truth',
                      color='black',
                      lw=2.5, ls='--', marker='s', markersize=8, markevery=2,
                      markerfacecolor='white')

    line2, = ax1.plot(baseline_forecasted[start_p:start_p + period],
                      label='Baseline', color='#C24841FF',
                      lw=2.5, marker='v', markersize=8, markevery=2,
                      markerfacecolor='white')

    line3, = ax1.plot(proposed_forecasted[start_p:start_p + period],
                      label='Proposed', color='#035AA6FF',
                      lw=2.5, marker='*', markersize=8, markevery=2,
                      markerfacecolor='white')


    ax1.set_ylabel('Norm. Load', fontsize=labelsize)
    #ax1.set_title('2018/02/24', fontsize=12)
    ax1.margins(x=0)

    ax2.plot(np.abs(true[start_p:start_p + period] - baseline_forecasted[start_p:start_p + period]),
             color='#C24841FF', lw=2.5, marker='v', markersize=8, markevery=2,
                      markerfacecolor='white')
    ax2.plot(np.abs(true[start_p:start_p + period] - proposed_forecasted[start_p:start_p + period]),
             color='#035AA6FF',
             lw=2.5, marker='*', markersize=8, markevery=2,
                      markerfacecolor='white')
    ax2.set_ylabel('Deviation', fontsize=labelsize)
    ax2.set_yticks([0, 0.1, 0.2])
    ax2.set_ylim(-0.05, 0.2)
    ax2.set_xlabel('Time Index [h]', fontsize=labelsize)
    ax2.margins(x=0)

    lines = [line1, line2, line3]  # 手动组合线条对象
    labels = [line.get_label() for line in lines]


    #start_p = 456 + 24 * 5
    #period = 24
    line1, = ax3.plot(true[start_p+24:start_p + period+24], label='Ground Truth',
                      color='black',
                      lw=2.5, ls='--', marker='s', markersize=8, markevery=2,
                      markerfacecolor='white')

    line2, = ax3.plot(baseline_forecasted[start_p+24:start_p + period+24],
                      label='Baseline', color='#C24841FF',
                      lw=2.5, marker='v', markersize=8, markevery=2,
                      markerfacecolor='white')

    line3, = ax3.plot(proposed_forecasted[start_p+24:start_p + period+24],
                      label='Proposed', color='#035AA6FF',
                      lw=2.5, marker='*', markersize=8, markevery=2,
                      markerfacecolor='white')

    #ax3.set_title('2018/02/25', fontsize=12)
    ax3.margins(x=0)

    ax4.plot(np.abs(true[start_p+24:start_p + period+24] -
                    baseline_forecasted[start_p+24:start_p + period+24]),
             color='#C24841FF', lw=2.5, marker='v', markersize=8, markevery=2,
             markerfacecolor='white')
    ax4.plot(np.abs(true[start_p+24:start_p + period+24] -
                    proposed_forecasted[start_p+24:start_p + period+24]),
             color='#035AA6FF',
             lw=2.5, marker='*', markersize=8, markevery=2,
             markerfacecolor='white')
    ax4.set_yticks([0, 0.1, 0.2])
    ax4.set_ylim(-0.05, 0.2)
    ax4.set_xlabel('Time Index [h]', fontsize=labelsize)
    ax4.margins(x=0)

    start_p = 456 + 24 * 7
    period = 24
    line1, = ax5.plot(true[start_p+48:start_p + period+48], label='Ground Truth',
                      color='black',
                      lw=2.5, ls='--', marker='s', markersize=5, markevery=2,
                      markerfacecolor='white')

    line2, = ax5.plot(baseline_forecasted[start_p+48:start_p + period+48],
                      label='Baseline', color='#C24841FF',
                      lw=2.5, marker='v', markersize=8, markevery=2,
                      markerfacecolor='white')

    line3, = ax5.plot(proposed_forecasted[start_p+48:start_p + period+48],
                      label='Proposed', color='#035AA6FF',
                      lw=2.5, marker='*', markersize=8, markevery=2,
                      markerfacecolor='white')

    #ax5.set_title('2018/02/27', fontsize=12)
    ax5.margins(x=0)

    ax6.plot(np.abs(true[start_p+48:start_p + period+48] -
                    baseline_forecasted[start_p+48:start_p + period+48]),
             color='#C24841FF', lw=2.5, marker='v', markersize=8, markevery=2,
             markerfacecolor='white')
    ax6.plot(np.abs(true[start_p+48:start_p + period+48] -
                    proposed_forecasted[start_p+48:start_p + period+48]),
             color='#035AA6FF',
             lw=2.5, marker='*', markersize=8, markevery=2,
             markerfacecolor='white')
    ax6.set_yticks([0, 0.1, 0.2])
    ax6.set_ylim(-0.05, 0.2)
    ax6.set_xlabel('Time Index [hour]', fontsize=ticksize)
    ax6.margins(x=0)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.95),  # 调整垂直位置
               ncol=4,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=False)

    ax1.tick_params(axis='both', labelsize=ticksize)
    ax2.tick_params(axis='both', labelsize=ticksize)
    ax3.tick_params(axis='both', labelsize=ticksize)
    ax4.tick_params(axis='both', labelsize=ticksize)
    ax5.tick_params(axis='both', labelsize=ticksize)
    ax6.tick_params(axis='both', labelsize=ticksize)

    plt.savefig('forecasting_curves.pdf')
    plt.show()

#plot_day_by_day()