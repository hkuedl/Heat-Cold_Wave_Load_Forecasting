import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from matplotlib import cm
import pypalettes
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from pyecharts import options as opts
from pyecharts.charts import Map
import random
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap




def basic_reduction(titlesize=16, ticksize=16, labelsize=16):
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey='row')

    ax = ax.flatten()

    colors = ['#46AEA0FF'] + ['#98D048FF'] + ['#F8D068FF'] + ['#88A0DCFF'] + \
             ['#F6B8BDFF'] + ['#F8A070FF']
    country_list = ["Guangdong", "PJM", "Texas", "India", "Hunan", "Europe"]

    mae_proposed_list = [0.02566, 0.034103405, 0.034255125, 0.02625, 0.02853, 0.024812333]
    rmse_proposed_list = [0.03378, 0.04681933, 0.043263125, 0.03537, 0.03705, 0.032820389]
    mae_baseline_list = [0.03344, 0.037433, 0.04211, 0.02914, 0.03227, 0.0331245]
    rmse_baseline_list = [0.04543, 0.051155579, 0.05253875, 0.03802, 0.04054, 0.042476944]


    bar_width = 0.5
    for i in range(6):
        # 计算数据
        data1 = [mae_proposed_list[i], rmse_proposed_list[i]]
        data2 = [mae_baseline_list[i] - mae_proposed_list[i],
                 rmse_baseline_list[i] - rmse_proposed_list[i]]

        # 设置 x 轴位置
        x = np.arange(len(data1))  # x 轴位置
        x_offset = 0.2  # 设置偏移量以减少柱子间距

        # 绘制柱状图
        bars1 = ax[i].bar(x - x_offset / 2, data1,
                          width=bar_width,
                          color=colors[i],
                          edgecolor='black',
                          linewidth=2,
                          label='Proposed')

        bars2 = ax[i].bar(x - x_offset / 2, data2,
                          width=bar_width,
                          bottom=data1,
                          color=colors[i], alpha=0.5,
                          edgecolor='black',
                          linestyle='--',
                          linewidth=2,  # 可选：添加填充图案
                          label='Reduction')

        # 计算并添加文本标签
        for j in range(len(data1)):
            if data2[j] != 0:  # 确保不除以零
                value = data2[j]/data1[j]
                ax[i].text(x[j] - x_offset / 2, data1[j] + data2[j] + 0.002, f'{value*100:.1f}'+'%↓',
                           ha='center', va='bottom', color='#C24841FF', fontsize=ticksize)

        # 美化坐标轴
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_ylim(0, 0.095)
        ax[i].set_yticks([0, 0.02, 0.04, 0.06, 0.08])
        ax[i].set_title(country_list[i], fontsize=titlesize)
        ax[i].legend(frameon=False, fontsize=ticksize)

        # 设置 x 轴范围以增加柱子与 y 轴之间的距离
        ax[i].set_xlim(-0.75, len(data1) - 0.5)

        # 设置 x 轴刻度和标签
        ax[i].set_xticks(x - x_offset / 2)  # 设置 x 轴刻度位置
        ax[i].set_xticklabels(['nMAE', 'nRMSE'], rotation=0, fontsize=ticksize, ha='center', va='top')  # 设置 x 轴标签
        ax[i].tick_params(axis='both', labelsize=labelsize)

    ax[0].set_ylabel('Forecasting Error', fontsize=labelsize)
    ax[3].set_ylabel('Forecasting Error', fontsize=labelsize)
    plt.tight_layout()

    plt.savefig('figures/overall_performance.pdf')
    plt.show()


#basic_reduction()

def plot_heatwave_map():

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Polygon

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    bmap = Basemap(llcrnrlon=115, llcrnrlat=23, urcrnrlon=121, urcrnrlat=29,
                   projection='lcc', lat_1=33, lat_2=45, lon_0=120, ax=ax1)
    shp_info = bmap.readshapefile('CHN_adm/CHN_adm3', 'states', drawbounds=False)

    for info, shp in zip(bmap.states_info, bmap.states):
        proid = info['NAME_1']
        if proid == 'Fujian':
            poly = Polygon(shp, facecolor='w', edgecolor='b', lw=0.2)
            ax1.add_patch(poly)

    bmap.drawcoastlines()
    bmap.drawcountries()
    bmap.drawparallels(np.arange(23, 29, 2), labels=[1, 0, 0, 0])
    bmap.drawmeridians(np.arange(115, 121, 2), labels=[0, 0, 0, 1])
    plt.title('Fujian Province')
    plt.savefig('fig_province.png', dpi=100, bbox_inches='tight')
    plt.clf()
    plt.close()

#plot_heatwave_map()


def plot_generated_sample_distribution_coldwave(titlesize=16, ticksize=12, labelsize=14, weather_type='coldwave'):
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
    # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots(1, 5, figsize=(18, 4))

    if weather_type == 'coldwave':
        fig.suptitle("Sample Distributions of 6 European Countries under Heat/Coldwaves",
                 fontsize=titlesize,
                 y=1)
        distribution_color = 'cadetblue'
    else:
        fig.suptitle("Sample Distributions of 6 European Countries under Common Periods",
                     fontsize=titlesize,
                     y=1)
        distribution_color = 'cornflowerblue'

    ax = ax.flatten()

    k = 0
    for country in ['Denmark', 'Finland', 'Germany', 'Norway', 'Sweden']:

        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2015/01/01/00')
        end_date = pd.to_datetime('2017/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

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
        for i in range(30, load.shape[0] // 24 - 3 - 30 - 6):
            ## load and temperature
            load_slice_list.append(load[24 * i:24 * (i + 8)])
            tem_slice_list.append(temperature[24 * i:24 * (i + 8)])

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

        coldwave_samples = generate_coldwave_samples(country, num_samples=400, weather_type=weather_type).cpu().detach().numpy()
        print(coldwave_samples[0, 0])

        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)
        print(coldwave_samples[0, 0])

        load_slice_list = np.array(load_slice_list)
        #print(load_slice_list.shape)
        tem_slice_list = np.array(tem_slice_list)

        sample_load_list = np.mean(load_slice_list[:, :-24], axis=1)
        label_load_list = np.mean(load_slice_list[:, -24:], axis=1)
        sample_tem_list = np.mean(tem_slice_list[:, :-24], axis=1)
        label_tem_list = np.mean(tem_slice_list[:, -24:], axis=1)
        generated_sample_load_list = np.mean(coldwave_samples[:, 0, :-24], axis=1)
        generated_sample_tem_list = np.mean(coldwave_samples[:, 1, :-24], axis=1)

        # set shared bins
        bins = np.linspace(
            min(min(sample_tem_list), min(generated_sample_tem_list)),
            max(max(sample_tem_list), max(generated_sample_tem_list)),
            50  # 分箱数量
        )

        # 绘制概率密度直方图（关键参数：density=True）
        ax[k].hist(sample_tem_list, bins=bins, density=True,
                 alpha=0.6, label='Original Samples', color='crimson')
        ax[k].hist(generated_sample_tem_list, bins=bins, density=True,
                 alpha=0.6, label='Generated Samples', color=distribution_color)

        # 添加核密度曲线（KDE）增强可视化
        kde_real = gaussian_kde(sample_tem_list)
        kde_gen = gaussian_kde(generated_sample_tem_list)
        x_vals = np.linspace(bins[0], bins[-1], 500)
        ax[k].plot(x_vals, kde_real(x_vals), color='darkblue', linewidth=2, linestyle='--', label='Original Samples Density')
        ax[k].plot(x_vals, kde_gen(x_vals), color='darkorange', linewidth=2, linestyle='--', label='Generated Samples Density')


        # 添加标注
        ax[k].set_xlabel('Norm. Temp.', fontsize=labelsize)
        ax[k].set_ylabel('Density', fontsize=labelsize)  # 纵轴标签改为概率密度
        ax[k].xaxis.set_tick_params(labelsize=ticksize)
        ax[k].yaxis.set_tick_params(labelsize=ticksize)
        ax[k].set_xlim(0, 1)
        ax[k].set_ylim(0, 5)
        ax[k].set_title(country, fontsize=titlesize)
        ax[k].spines['top'].set_visible(False)
        ax[k].spines['right'].set_visible(False)

        k += 1
        #plt.legend()
        #plt.grid(True, linestyle='--', alpha=0.5)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.93),  # 调整垂直位置
               ncol=4,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=False)

    #fig.subplots_adjust(top=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    if weather_type == 'coldwave':
        plt.savefig('figures/sample_distribution_coldwave.pdf')
    else:
        plt.savefig('figures/sample_distribution_common.pdf')

    plt.show()





#plot_generated_sample_distribution_coldwave(weather_type='coldwave')

#plot_generated_sample_distribution_coldwave(weather_type='common')



def plot_generated_interval_coldwave(weather_type='common', titlesize=13, ticksize=13, labelsize=13):
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
    #from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe
    color = '#4D97CD'
    color2 = '#DB6968'


    for country in ['Norway']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2015/01/01/00')
        end_date = pd.to_datetime('2017/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

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
        for i in range(30, load.shape[0]//192 - 3):
            ## load and temperature
            load_slice_list.append(load[192 * i:192 * i + 192])
            tem_slice_list.append(temperature[192 * i:192 * i + 192])

        coldwave_samples = generate_coldwave_samples(country, num_samples=200, weather_type=weather_type).cpu().detach().numpy()
        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)

        avg_coldwave_load = np.mean(coldwave_samples, axis=0)
        avg_coldwave_tem = np.mean(coldwave_samples, axis=0)

        load_slice_list = np.array(load_slice_list)
        tem_slice_list = np.array(tem_slice_list)


        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        #fig, ax = plt.subplots(2, 4, figsize=(18, 6))
        fig, ax = plt.subplots(2, 8, figsize=(20, 6),
                                 #sharex=True,  # 所有子图共享x轴
                                 #gridspec_kw={'hspace': 0.00})  # 行间无间距
                               )
        #ax = ax.flatten()
        # 计算均值、最大值和最小值
        mean_load = np.mean(load_slice_list, axis=0)
        max_load = np.max(load_slice_list, axis=0)
        min_load = np.min(load_slice_list, axis=0)

        mean_tem = np.mean(tem_slice_list, axis=0)
        max_tem = np.max(tem_slice_list, axis=0)
        min_tem = np.min(tem_slice_list, axis=0)
        x = np.arange(192)


        # 绘制均值曲线
        #ax[0].plot(x, mean_curve, color='blue', label='Mean Curve')

        for j in range(8):
            line1 = ax[0, j].fill_between(x[24*j:24*(j+1)], min_load[24*j:24*(j+1)],
                                          max_load[24*j:24*(j+1)], color='silver', alpha=0.2,
                                          label='Original Samples')

            ax[1, j].fill_between(x[24 * j:24 * (j + 1)], min_tem[24 * j:24 * (j + 1)],
                                          max_tem[24 * j:24 * (j + 1)], color='silver', alpha=0.2,
                                          )

            line2, = ax[0, j].plot(x[24*j:24*(j+1)], coldwave_samples[0, 0, :][24*j:24*(j+1)],
                                   color=color, label='Synthetic Load', lw=1)

            line3, = ax[1, j].plot(x[24 * j:24 * (j + 1)], coldwave_samples[0, 1, :][24 * j:24 * (j + 1)],
                                   color=color2, label = 'Synthetic Temp.',
                        lw=1)

            line4, = ax[0, j].plot(x[24 * j:24 * (j + 1)], avg_coldwave_load[0, :][24 * j:24 * (j + 1)], color='blue',
                          lw=2, marker='o', markevery=3,
                                   label='Synthetic Avg. Load',
                                   markerfacecolor='white')

            line5, = ax[1, j].plot(x[24 * j:24 * (j + 1)], avg_coldwave_tem[1, :][24 * j:24 * (j + 1)], color='crimson',
                                   lw=2, marker='o', markevery=3,
                                   label='Synthetic Avg. Temp.',
                                   markerfacecolor='white')

            for k in range(65, 200-1):
                ax[0, j].plot(x[24 * j:24 * (j + 1)], coldwave_samples[k+1, 0, :][24 * j:24 * (j + 1)], color=color, lw=0.1)
                ax[1, j].plot(x[24 * j:24 * (j + 1)], coldwave_samples[k + 1, 1, :][24 * j:24 * (j + 1)], color=color2,
                              lw=0.1)
            #ax[0, j].plot(x[24 * j:24 * (j + 1)], max_load[24 * j:24 * (j + 1)],
            #           color='orangered', ls='-.', label='Envelopes for Original Samples')
            #ax[0, j].plot(x[24 * j:24 * (j + 1)], min_load[24 * j:24 * (j + 1)],
            #           color='orangered', ls='-.')




            #for k in range(0, 24):
            #    box = ax[j].boxplot(coldwave_samples[:, 0, 24 * j+k],
            #                      positions=[24*j+k],
            #                      patch_artist=True,  # 允许填充颜色
            #                      widths=0,  # 箱型图宽度
            #                      showmeans=False,
            #                      showfliers=False,
            #                      medianprops={'visible': False})  # 时间步标签（1-24）
                #for b in box['boxes']:
                #    b.set(facecolor='none', edgecolor='none', linewidth=0)  # 完全透明

                #for element in ['boxes', 'whiskers', 'caps', 'medians']:
                #    plt.setp(box[element], color=color)  # 将边框、须线、端帽、中位线设为红色


            #ax[0, j].plot(x[24 * j:24 * (j + 1)],
            #           np.mean(coldwave_samples[:, 0, 24 * j:24 * (j + 1)], axis=0),
            #           color=color, marker='o', ls='-',
            #           markerfacecolor='white', label='Generated Sample')

            #ax[1, j].plot(x[24 * j:24 * (j + 1)],
            #           np.mean(coldwave_samples[:, 1, 24 * j:24 * (j + 1)], axis=0),
            #           color=color, marker='o', ls='-',
            #           markerfacecolor='white', label='Generated Sample')

            #ax[0, j].spines['top'].set_visible(False)
            #ax[0, j].spines['right'].set_visible(False)
            ax[0, j].set_ylim(0, 1)
            ax[0, j].xaxis.set_tick_params(labelsize=ticksize)
            ax[0, j].yaxis.set_tick_params(labelsize=ticksize)

            #ax[1, j].spines['top'].set_visible(False)
            #ax[1, j].spines['right'].set_visible(False)
            ax[1, j].set_ylim(0, 1)
            ax[1, j].xaxis.set_tick_params(labelsize=ticksize)
            ax[1, j].yaxis.set_tick_params(labelsize=ticksize)

            if j != 7:
                ax[0, j].set_title('Day {}'.format(j+1), fontsize=titlesize)
            else:
                ax[0, j].set_title('Forecasted Day'.format(j + 1), fontsize=titlesize)

            #if j == 0 or j == 4:

            if j == 0:
                ax[0, j].set_ylabel('Norm. Load', fontsize=labelsize)
                ax[1, j].set_ylabel('Norm. Temp.', fontsize=labelsize)
                ax[0, j].set_yticks([0, 0.5])
                ax[1, j].set_yticks([0, 0.5])
            else:
                ax[0, j].set_yticks([0, 0.5])
                ax[1, j].set_yticks([0, 0.5])
                ax[0, j].set_yticklabels([])
                ax[1, j].set_yticklabels([])
            #ax[j+8].set_xlabel('Time Index [hour]', fontsize=labelsize)
            ax[0, j].set_xticks([24*j, 24*j+12,  24*(j+1)])
            ax[1, j].set_xticks([24*j, 24*j+12,  24*(j+1)])
            ax[0, j].set_xticklabels([])
            ax[1, j].set_xticklabels([24*j, 24*j+12,  24*(j+1)])

            #ax[0, j].set_xlim(24 * j, 24 * (j + 1))
            #ax[1, j].set_xlim(24 * j, 24 * (j + 1))
            ax[0, j].spines['top'].set_visible(False)
            ax[0, j].spines['right'].set_visible(False)
            ax[1, j].spines['top'].set_visible(False)
            ax[1, j].spines['right'].set_visible(False)


        # 合并两个子图的图例句柄和标签
        handles = [line1, line2, line3]  # 所有线条的句柄
        labels = [h.get_label() for h in handles]  # 对应的标签

        # 在图像底部添加共同图例
        #fig.legend(
        #    handles=handles,
        #    labels=labels,
        #    loc="lower center",  # 图例位置（底部居中）
        #    bbox_to_anchor=(0.5, 0),  # 调整位置（x, y偏移）
        #    ncol=2,  # 图例列数（根据条目数量调整）
        #    frameon=True,  # 是否显示边框
        #    fontsize=15,
        #    edgecolor='black'
        #)

        #ax[0].set_ylabel('Normalized Load', fontsize=14)
        #ax[1].set_ylabel('Normalized Temperature', fontsize=14)
        #ax[1].set_xlabel('Time Index [h]', fontsize=14)

        #ax[0].tick_params(axis='both', labelsize=14)
        #ax[1].tick_params(axis='both', labelsize=14)



        #plt.tight_layout()
        #fig.suptitle("Load Samples Visualization",
        #             fontsize=titlesize,
        #             y=1)
        #fig.suptitle("Time Index [hour]",
        #             fontsize=titlesize,
        #             y=0.05)
        fig.supxlabel('Time Index [hour]', fontsize=labelsize, y=0.02)  # y参数控制垂直位置
        #handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.95),  # 调整垂直位置
                   ncol=4,  # 设置列数以水平排列
                   fontsize=ticksize,
                   frameon=False)
        #fig.subplots_adjust(bottom=0.2, top=0.9)  # 预留底部空间
        #ax[0].set_title('Sample Visualization', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.87])

        plt.savefig('figures/synthetic_sample_vis.pdf')
        plt.show()


#plot_generated_interval_coldwave('coldwave')



def plot_generated_interval_coldwave_2(weather_type='common', titlesize=18, ticksize=14, labelsize=18):
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe
    color = '#4D97CD'
    color2 = '#DB6968'
    color3 = '#33ABC1'
    color4 = 'crimson'


    for country in ['Norway']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2015/01/01/00')
        end_date = pd.to_datetime('2017/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

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
        extreme_load_list = []
        extreme_tem_list = []

        for i in range(30, load.shape[0]//192):
            ## load and temperature
            load_slice_list.append(load[192 * i:192 * i + 192])
            tem_slice_list.append(temperature[192 * i:192 * i + 192])

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[8 * i+7:8 * i + 10]) - T_05
            ECI_accl = np.mean(T_i_list[8 * i+7:8 * i + 10]) - np.mean(T_i_list[8 * i +7:8 * i+10])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                extreme_load_list.append(load[192 * i:192 * i+192])
                extreme_tem_list.append(temperature[192 * i:192 * i+192])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[i:i + 3]) - T_95
            EHI_accl = np.mean(T_i_list[i:i + 3]) - np.mean(T_i_list[i - 30:i])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

        coldwave_samples = generate_coldwave_samples(country, num_samples=200, weather_type=weather_type).cpu().detach().numpy()
        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)

        avg_coldwave_load = np.mean(coldwave_samples, axis=0)
        avg_coldwave_tem = np.mean(coldwave_samples, axis=0)

        load_slice_list = np.array(load_slice_list)
        tem_slice_list = np.array(tem_slice_list)
        extreme_load_list = np.array(extreme_load_list)
        extreme_tem_list = np.array(extreme_tem_list)


        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        #fig, ax = plt.subplots(2, 4, figsize=(18, 6))
        fig, ax = plt.subplots(2, 2, figsize=(9, 6))
        #ax = ax.flatten()

        mean_load = np.mean(load_slice_list, axis=0)
        max_load = np.max(load_slice_list, axis=0)
        min_load = np.min(load_slice_list, axis=0)

        mean_tem = np.mean(tem_slice_list, axis=0)
        max_tem = np.max(tem_slice_list, axis=0)
        min_tem = np.min(tem_slice_list, axis=0)
        x = np.arange(192)




        title = ['Input Sequences', 'Labels']

        ## 前两张图分别画
        for j in range(0, 2):
            # Define index mapping: j=0 corresponds to the range (0, 24), j=1 corresponds to the range (168, 192).
            if j == 0:
                start_idx = 72
                end_idx = 96
            elif j == 1:
                start_idx = 168
                end_idx = 192

            line1 = ax[0, j].fill_between(x[start_idx:end_idx], min_load[start_idx:end_idx],
                                          max_load[start_idx:end_idx], color='silver', alpha=0.2,
                                          label='Original Samples')

            ax[1, j].fill_between(x[start_idx:end_idx], min_tem[start_idx:end_idx],
                                  max_tem[start_idx:end_idx], color='silver', alpha=0.2)

            line4, = ax[0, j].plot(x[start_idx:end_idx], extreme_load_list[0][start_idx:end_idx],
                                   color=color3, label='Extreme Load', lw=1)

            line5, = ax[1, j].plot(x[start_idx:end_idx], extreme_tem_list[0][start_idx:end_idx],
                                   color=color4, label='Extreme Temp.', lw=1)

            divider = make_axes_locatable(ax[0, j])

            margin_ax = divider.append_axes("right", size="20%", pad=0.1)

            # Calculate the average of all samples within the specified interval.
            load_means = [np.mean(sample[start_idx:end_idx]) for sample in extreme_load_list]
            n, bins, patches = margin_ax.hist(load_means, bins=60, orientation='horizontal',
                                              alpha=0.7, color='royalblue', density=True)
            margin_ax.set_xticks([])
            margin_ax.set_yticks([])
            margin_ax.set_ylim(0, 1)

            divider = make_axes_locatable(ax[1, j])

            margin_ax = divider.append_axes("right", size="20%", pad=0.1)

            # Calculate the average of all samples within the specified interval.
            tem_means = [np.mean(sample[start_idx:end_idx]) for sample in extreme_tem_list]
            n, bins, patches = margin_ax.hist(tem_means, bins=30, orientation='horizontal',
                                              alpha=0.3, color='#AE2012', density=True)
            margin_ax.set_xticks([])
            margin_ax.set_yticks([])
            margin_ax.set_ylim(0, 1)

            for k in range(len(extreme_tem_list)-1):
                ax[0, j].plot(x[start_idx:end_idx], extreme_load_list[k + 1][start_idx:end_idx],
                              color=color3, lw=1)
                ax[1, j].plot(x[start_idx:end_idx], extreme_tem_list[k + 1][start_idx:end_idx],
                              color=color4, lw=1)

            ax[0, j].set_ylim(0, 1)
            ax[0, j].xaxis.set_tick_params(labelsize=ticksize)
            ax[0, j].yaxis.set_tick_params(labelsize=ticksize)

            ax[1, j].set_ylim(0, 1)
            ax[1, j].xaxis.set_tick_params(labelsize=ticksize)
            ax[1, j].yaxis.set_tick_params(labelsize=ticksize)

            if j != 7:
                ax[0, j].set_title(title[j], fontsize=titlesize)
            else:
                ax[0, j].set_title('Forecasted Day', fontsize=titlesize)

            if j == 0:
                ax[0, j].set_ylabel('Norm. Load', fontsize=labelsize)
                ax[1, j].set_ylabel('Norm. Temp.', fontsize=labelsize)
                ax[0, j].set_yticks([0, 0.5])
                ax[1, j].set_yticks([0, 0.5])
            else:
                ax[0, j].set_yticks([0, 0.5])
                ax[1, j].set_yticks([0, 0.5])
                ax[0, j].set_yticklabels([])
                ax[1, j].set_yticklabels([])

            # change x-axis ticks and labels
            ax[0, j].set_xticks([start_idx, start_idx + 12, end_idx])
            ax[1, j].set_xticks([start_idx, start_idx + 12, end_idx])
            ax[0, j].set_xticklabels([])
            ax[1, j].set_xticklabels([start_idx, start_idx + 12, end_idx])

            ax[0, j].spines['top'].set_visible(False)
            ax[0, j].spines['right'].set_visible(False)
            ax[1, j].spines['top'].set_visible(False)
            ax[1, j].spines['right'].set_visible(False)


        # 合并两个子图的图例句柄和标签
        handles = [line1, line4, line5]  # 所有线条的句柄
        labels = [h.get_label() for h in handles]  # 对应的标签


        fig.supxlabel('Time Index [hour]', fontsize=labelsize, y=0.02)  # y参数控制垂直位置
        #handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.94),  # 调整垂直位置
                   ncol=3,  # 设置列数以水平排列
                   fontsize=ticksize,
                   frameon=True,
                   edgecolor='black')

        fig.suptitle("Input Sequences and Labels in the Original Dataset",
                     y=1, fontsize=titlesize)
        #fig.subplots_adjust(bottom=0.2, top=0.9)  # 预留底部空间
        #ax[0].set_title('Sample Visualization', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        plt.savefig('figures/synthetic_sample_vis.pdf')
        plt.show()



def plot_original_interval_coldwave_2(weather_type='common', titlesize=18, ticksize=14, labelsize=18):
        import sys
        sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
        sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
        sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
        sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
        print(sys.path)
        from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
        # Europe
        color = '#4D97CD'
        color2 = '#DB6968'
        color3 = '#33ABC1'
        color4 = 'crimson'

        for country in ['Norway']:
            # country = 'COAST'
            data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
            start_date = pd.to_datetime('2015/01/01/00')
            end_date = pd.to_datetime('2017/12/31/23')
            data = data[
                (pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

            load = np.array(data['Load'])
            load = (load - min(load)) / (max(load) - min(load))
            temperature = np.array(data['Temperature'])
            temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

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
            extreme_load_list = []
            extreme_tem_list = []

            for i in range(30, load.shape[0] // 192):
                ## load and temperature
                load_slice_list.append(load[192 * i:192 * i + 192])
                tem_slice_list.append(temperature[192 * i:192 * i + 192])

                ## define the cold wave index
                ECI_sig = np.mean(T_i_list[8 * i + 7:8 * i + 10]) - T_05
                ECI_accl = np.mean(T_i_list[8 * i + 7:8 * i + 10]) - np.mean(T_i_list[8 * i + 7:8 * i + 10])
                ECF = min(0, -ECI_sig * min(-1, ECI_accl))
                coldwave_index.append(float(ECF < 0))
                if ECF < 0:
                    extreme_load_list.append(load[192 * i:192 * i + 192])
                    extreme_tem_list.append(temperature[192 * i:192 * i + 192])

                ## define the hot wave index
                EHI_sig = np.mean(T_i_list[i:i + 3]) - T_95
                EHI_accl = np.mean(T_i_list[i:i + 3]) - np.mean(T_i_list[i - 30:i])
                EHF = max(0, EHI_sig * max(1, EHI_accl))
                hotwave_index.append(float(EHF > 0))

            coldwave_samples = generate_coldwave_samples(country, num_samples=200,
                                                         weather_type=weather_type).cpu().detach().numpy()
            coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                        coldwave_samples.shape[1], -1)

            avg_coldwave_load = np.mean(coldwave_samples, axis=0)
            avg_coldwave_tem = np.mean(coldwave_samples, axis=0)

            load_slice_list = np.array(load_slice_list)
            tem_slice_list = np.array(tem_slice_list)
            extreme_load_list = np.array(extreme_load_list)
            extreme_tem_list = np.array(extreme_tem_list)

            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            # fig, ax = plt.subplots(2, 4, figsize=(18, 6))
            fig, ax = plt.subplots(2, 2, figsize=(9, 6))
            # ax = ax.flatten()
            # 计算均值、最大值和最小值
            mean_load = np.mean(load_slice_list, axis=0)
            max_load = np.max(load_slice_list, axis=0)
            min_load = np.min(load_slice_list, axis=0)

            mean_tem = np.mean(tem_slice_list, axis=0)
            max_tem = np.max(tem_slice_list, axis=0)
            min_tem = np.min(tem_slice_list, axis=0)
            x = np.arange(192)

            # 绘制均值曲线
            # ax[0].plot(x, mean_curve, color='blue', label='Mean Curve')

            title = ['Input Sequences', 'Labels']

            ## 前两张图分别画

            for j in range(0, 2):

                if j == 0:
                    start_idx = 72
                    end_idx = 96
                elif j == 1:
                    start_idx = 168
                    end_idx = 192

                # 定义索引映射：j=2对应区间(0,24)，j=3对应区间(168,192)
                if j == 2:
                    start_idx = 72
                    end_idx = 96
                elif j == 3:
                    start_idx = 168
                    end_idx = 192

                line1 = ax[0, j].fill_between(x[start_idx:end_idx], min_load[start_idx:end_idx],
                                              max_load[start_idx:end_idx], color='silver', alpha=0.2,
                                              label='Original Samples')

                ax[1, j].fill_between(x[start_idx:end_idx], min_tem[start_idx:end_idx],
                                      max_tem[start_idx:end_idx], color='silver', alpha=0.2)

                line2, = ax[0, j].plot(x[start_idx:end_idx], coldwave_samples[0, 0, :][start_idx:end_idx],
                                       color=color, label='Extreme Load', lw=1)

                line3, = ax[1, j].plot(x[start_idx:end_idx], coldwave_samples[0, 1, :][start_idx:end_idx],
                                       color=color2, label='Extreme Temp.', lw=1)

                divider = make_axes_locatable(ax[0, j])

                margin_ax = divider.append_axes("right", size="20%", pad=0.1)
                n, bins, patches = margin_ax.hist(np.mean(coldwave_samples[:, 0, start_idx:end_idx], axis=1),
                                                  bins=60, orientation='horizontal',
                                                  alpha=0.7, color='royalblue', density=True)
                margin_ax.set_xticks([])
                margin_ax.set_yticks([])
                margin_ax.set_ylim(0, 1)

                divider = make_axes_locatable(ax[1, j])

                margin_ax = divider.append_axes("right", size="20%", pad=0.1)
                n, bins, patches = margin_ax.hist(np.mean(coldwave_samples[:, 1, start_idx:end_idx], axis=1),
                                                  bins=30, orientation='horizontal',
                                                  alpha=0.3, color='#AE2012', density=True)
                margin_ax.set_xticks([])
                margin_ax.set_yticks([])
                margin_ax.set_ylim(0, 1)

                for k in range(65, 200 - 1):
                    ax[0, j].plot(x[start_idx:end_idx], coldwave_samples[k + 1, 0, :][start_idx:end_idx],
                                  color=color, lw=0.15)
                    ax[1, j].plot(x[start_idx:end_idx], coldwave_samples[k + 1, 1, :][start_idx:end_idx],
                                  color=color2, lw=0.15)

                ax[0, j].set_ylim(0, 1)
                ax[0, j].xaxis.set_tick_params(labelsize=ticksize)
                ax[0, j].yaxis.set_tick_params(labelsize=ticksize)

                ax[1, j].set_ylim(0, 1)
                ax[1, j].xaxis.set_tick_params(labelsize=ticksize)
                ax[1, j].yaxis.set_tick_params(labelsize=ticksize)

                if j != 7:
                    ax[0, j].set_title(title[j - 2], fontsize=titlesize)
                else:
                    ax[0, j].set_title('Labels', fontsize=titlesize)

                if j == 0:
                    ax[0, j].set_ylabel('Norm. Load', fontsize=labelsize)
                    ax[1, j].set_ylabel('Norm. Temp.', fontsize=labelsize)
                    ax[0, j].set_yticks([0, 0.5])
                    ax[1, j].set_yticks([0, 0.5])
                else:
                    ax[0, j].set_yticks([0, 0.5])
                    ax[1, j].set_yticks([0, 0.5])
                    ax[0, j].set_yticklabels([])
                    ax[1, j].set_yticklabels([])

                # 修改x轴刻度标签以反映实际的时间索引
                ax[0, j].set_xticks([start_idx, start_idx + 12, end_idx])
                ax[1, j].set_xticks([start_idx, start_idx + 12, end_idx])
                ax[0, j].set_xticklabels([])
                ax[1, j].set_xticklabels([start_idx, start_idx + 12, end_idx])

                ax[0, j].spines['top'].set_visible(False)
                ax[0, j].spines['right'].set_visible(False)
                ax[1, j].spines['top'].set_visible(False)
                ax[1, j].spines['right'].set_visible(False)

            # 合并两个子图的图例句柄和标签
            handles = [line1, line2, line3]  # 所有线条的句柄
            labels = [h.get_label() for h in handles]  # 对应的标签

            fig.supxlabel('Time Index [hour]', fontsize=labelsize, y=0.02)  # y参数控制垂直位置
            # handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(0.5, 0.94),  # 调整垂直位置
                       ncol=3,  # 设置列数以水平排列
                       fontsize=ticksize,
                       frameon=True,
                       edgecolor='black')
            fig.suptitle("Input Sequences and labels in the Synthetic Dataset",
                         y=1, fontsize=titlesize)
            # fig.subplots_adjust(bottom=0.2, top=0.9)  # 预留底部空间
            # ax[0].set_title('Sample Visualization', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.92])

            plt.savefig('figures/synthetic_sample_vis.pdf')
            plt.show()


#plot_generated_interval_coldwave_2('coldwave')
#plot_generated_interval_coldwave_2('common')
#plot_original_interval_coldwave_2('common')
#plot_original_interval_coldwave_2('coldwave')


def plot_guangdong_map(titlesize=16, ticksize=14, labelsize=14, region='guangdong'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


    city_name = ['chaozhou', 'dongguan', 'foshan', 'guangzhou', 'heyuan',
                 'huizhou', 'jiangmen', 'jieyang', 'maoming', 'meizhou',
                 'qingyuan', 'shantou', 'shanwei', 'shaoguan', 'shenzhen',
                 'yangjiang', 'yunfu', 'zhanjiang', 'zhaoqing', 'zhongshan',
                 'zhuhai'
                 ]

    city_code = ['445100', '441900', '440600', '440100', '441600',
                 '441300', '440700', '445200', '440900', '441400',
                 '441800', '440500', '441500', '440200', '440300',
                 '441700', '445300', '440800', '441200', '442000',
                 '440400']
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)


    cmap = plt.get_cmap('GnBu')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
    common_tem_norm = []
    hotwave_tem_norm = []
    coldwave_tem_norm = []
    for i in range(len(city_code)):
        data = pd.read_excel('../Data/reformed_data_updated/GuangDong_data_reformed/{}.xlsx'.format(city_name[i]))

        strat_time = '2020/01/01/00'
        end_time = '2022/12/31/23'
        start_date = pd.to_datetime(strat_time)  ## Thursday
        end_date = pd.to_datetime(end_time)

        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]
        data['Data_Hour'] = pd.to_datetime(data['Data_Hour'])  # 确保 Data_Hour 列为 datetime 类型

        load = np.array(data['Load'])
        temperature = np.array(data['Temperature'])

        norm_load = (load - min(load)) / (max(load) - min(load))
        norm_tem = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])
        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)

        # load and temperature slices formulation
        coldwave_index = []
        hotwave_index = []
        common_load_mean = []
        hotwave_load_mean = []
        coldwave_load_mean = []


        for j in range(30, load.shape[0] // 24 - 3 - 30 - 6):

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[j:j + 3]) - T_05
            ECI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                coldwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                coldwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                coldwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[j:j + 3]) - T_95
            EHI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

            if EHF > 0:
                hotwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                hotwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                hotwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

            if ECF == 0 and EHF == 0:
                common_load_mean.append(load[24 * j: 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])


        hot_common_ratio = np.mean(hotwave_load_mean)/np.mean(common_load_mean)-1
        color = cmap(norm(hot_common_ratio))



        print(i)
        guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        #loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.65, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(hotwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')

    ax_inset_2 = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.15, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )



    bins = np.linspace(
        0,
        1,
        100  # 分箱数量
    )

    # 计算两个分布的直方图数据(np.mean(np.array(hotwave_tem_norm), axis=1)
    counts_hotwave, bin_edges = np.histogram(np.mean(np.array(hotwave_tem_norm), axis=1), bins=bins)
    counts_common, _ = np.histogram(np.mean(np.array(common_tem_norm), axis=1), bins=bin_edges)  # 使用相同的bin_edges

    ax_inset_2.bar(
        x=bin_edges[:-1],
        height=counts_common,
        width=np.diff(bin_edges),
        # bottom=counts_hotwave,  # 关键参数：堆叠在common的柱子上方
        align='edge',
        color='grey',
        alpha=0.7,
        linewidth=1,
        label='Hotwave Scenario'
    )

    ax_inset_2.bar(
      x=bin_edges[:-1],  # 分箱左边界
      height=counts_hotwave,
      width=np.diff(bin_edges),  # 分箱宽度
      align='edge',  # 柱子对齐分箱边缘
      color='salmon',
      alpha=1,
      linewidth=1,
      label='Common Scenario'
      )



    ax_inset_2.set_xlabel('Norm. Tem.', fontsize=labelsize)
    ax_inset_2.set_ylabel('Density', fontsize=labelsize)
    ax_inset_2.set_yticks([])
    ax_inset_2.set_title('Sample Distribution', fontsize=labelsize)
    ax_inset_2.xaxis.set_tick_params(labelsize=ticksize)



    ax.set_yticks([20, 22, 24, 26])
    ax.set_xticks([110, 112, 114, 116, 118])
    ax.set_xticklabels(['110°E', '112°E', '114°E', '116°E', '118°E'])
    ax.set_yticklabels(['20°N', '22°N', '24°N', '26°N'])




    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Pattern', fontsize=labelsize)
    ax_inset.set_xlabel('Time Index [hour]', fontsize=labelsize)
    ax_inset.set_ylabel('Norm. Load', fontsize=labelsize)


    ax.set_xlim(109, 119)
    ax.set_ylim(19, 26)
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(0.8)

    # 设置标题和显示
    ax.set_title("Heatwave in Guangdong", fontsize=titlesize)
    plt.tight_layout()

    #plt.axis('equal')
    #plt.show()

    fig.savefig('figures/load_change_in_{}.pdf'.format(region))
    fig.show()


def plot_hunan_map(titlesize=16, ticksize=14, labelsize=14, region='hunan'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path

    city_name = ['娄底', '岳阳', '常德', '张家界', '怀化', '株洲',
                 '永州', '湘潭', '湘西', '益阳', '衡阳', '邵阳', '郴州', '长沙']

    city_code = ['431300', '430600', '430700', '430800', '431200',
                 '430200', '431100', '430300', '433100', '430900',
                 '430400', '430500', '431000', '430100']

    fig, ax = plt.subplots(figsize=(6, 6))
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)


    cmap = plt.get_cmap('RdPu')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
    coldwave_tem_norm = []
    common_tem_norm = []
    for i in range(len(city_code)):
        data = pd.read_csv('../Data/reformed_data_updated/hunan_data_reformed/{}.csv'.format(city_name[i]), header=0, usecols=['Date', 'load', 'temp'])

        strat_time = '2021/01/01/00'
        end_time = '2023/10/30/23'
        start_date = pd.to_datetime(strat_time)  ## Thursday
        end_date = pd.to_datetime(end_time)

        data = data[(pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)]

        data['Date'] = pd.to_datetime(data['Date'])  # 确保 Data_Hour 列为 datetime 类型
        # data['Is_Weekend'] = data['Data_Hour'].dt.dayofweek >= 5  # 0=周一, 1=周二, ..., 6=周日
        # data['Is_Holiday'] = data['Data_Hour'].dt.date.isin(pd.to_datetime(['2015-01-01', '2015-12-25', '2016-01-01', '2016-12-25', ...]).date)  # 添加你的节假日列表
        load = np.array(data['load'])
        temperature = np.array(data['temp'])


        norm_load = (load-min(load))/(max(load)-min(load))
        norm_tem = (temperature - min(temperature)) / (max(temperature) - min(temperature))



        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])
        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)

        # load and temperature slices formulation
        coldwave_index = []
        hotwave_index = []
        common_load_mean = []
        hotwave_load_mean = []
        coldwave_load_mean = []


        for j in range(30, load.shape[0] // 24 - 3 - 30 - 6):

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[j:j + 3]) - T_05
            ECI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                coldwave_load_mean.append(load[24 * j : 24 * (j + 1)])
                coldwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                coldwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[j:j + 3]) - T_95
            EHI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

            if EHF > 0:
                hotwave_load_mean.append(load[24 * j : 24 * (j + 1)])
                hotwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])

            if ECF == 0 and EHF == 0:
                common_load_mean.append(load[24 * j : 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

        hot_common_ratio = np.mean(hotwave_load_mean)/np.mean(common_load_mean)-1
        color = cmap(norm(hot_common_ratio))

        print(i)
        guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.65, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(hotwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')

    ax_inset_2 = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.15, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    bins = np.linspace(
        0,
        1,
        100  # 分箱数量
    )

    # 计算两个分布的直方图数据(np.mean(np.array(hotwave_tem_norm), axis=1)
    counts_hotwave, bin_edges = np.histogram(np.mean(np.array(coldwave_tem_norm), axis=1), bins=bins)
    counts_common, _ = np.histogram(np.mean(np.array(common_tem_norm), axis=1), bins=bin_edges)  # 使用相同的bin_edges

    ax_inset_2.bar(
        x=bin_edges[:-1],
        height=counts_common,
        width=np.diff(bin_edges),
        # bottom=counts_hotwave,  # 关键参数：堆叠在common的柱子上方
        align='edge',
        color='grey',
        alpha=0.7,
        linewidth=1,
        label='Hotwave Scenario'
    )

    ax_inset_2.bar(
        x=bin_edges[:-1],  # 分箱左边界
        height=counts_hotwave,
        width=np.diff(bin_edges),  # 分箱宽度
        align='edge',  # 柱子对齐分箱边缘
        color='salmon',
        alpha=1,
        linewidth=1,
        label='Common Scenario'
    )

    ax_inset_2.set_xlabel('Norm. Tem.', fontsize=labelsize)
    ax_inset_2.set_ylabel('Density', fontsize=labelsize)
    ax_inset_2.set_yticks([])
    ax_inset_2.set_title('Sample Distribution', fontsize=labelsize)
    ax_inset_2.xaxis.set_tick_params(labelsize=ticksize)


    ax.set_yticks([24, 26, 28, 30, 32, 34])
    ax.set_xticks([108, 110, 112, 114, 116, 118])
    ax.set_xticklabels(['108°E', '110°E', '112°E', '114°E', '116°E', '118°E'])
    ax.set_yticklabels(['24°N', '26°N', '28°N', '30°N', '32°N', '34°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Pattern', fontsize=labelsize)
    ax_inset.set_xlabel('Time Index [hour]', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_xlim(108, 118)
    ax.set_ylim(23, 30)
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(0.8)

    # 设置标题和显示
    ax.set_title("Coldwave in Hunan", fontsize=titlesize)
    plt.tight_layout()
    fig.savefig('figures/load_change_in_{}.pdf'.format(region))
    #plt.axis('equal')
    plt.show()


def plot_europe_map(titlesize=16, ticksize=14, labelsize=14, region='europe'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path

    city_name = ['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland',
                      'Great_Britain', 'Estonia', 'Greece', 'Luxembourg',
                      'Spain', 'Slovakia', 'Bulgaria', 'Czech Republic']

    #city_name = ['Czech Republic']


    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    print(world[world['NAME'].str.contains('Czechia', case=False)])

    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)


    cmap = plt.get_cmap('Oranges')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
    coldwave_tem_norm = []
    common_tem_norm = []
    for i in range(len(city_name)):
        if city_name[i] == 'Great_Britain':
            country = 'United Kingdom'
        elif city_name[i] == 'Czech Republic':
            country = 'Czechia'
        else:
            country = city_name[i]

        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(city_name[i]))

        strat_time = '2015/01/01/00'
        end_time = '2017/12/31/23'
        start_date = pd.to_datetime(strat_time)  ## Thursday
        end_date = pd.to_datetime(end_time)

        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        temperature = np.array(data['Temperature'])


        norm_load = (load-min(load))/(max(load)-min(load))
        norm_tem = (temperature - min(temperature)) / (max(temperature) - min(temperature))



        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])
        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)

        #print(T_05)

        # load and temperature slices formulation
        coldwave_index = []
        hotwave_index = []
        common_load_mean = []
        hotwave_load_mean = []
        coldwave_load_mean = []

        for j in range(30, load.shape[0] // 24 - 3 - 30 - 6):
            if np.any(np.isnan(load[24 * j: 24 * (j + 1)])) or np.any(np.isnan(temperature[24 * i: 24 * (i + 1)])):
                continue  # 跳过当前循环

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[j:j + 3]) - T_05
            ECI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                coldwave_load_mean.append(load[24 * j : 24 * (j + 1)])
                coldwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                coldwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])
            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[j:j + 3]) - T_95
            EHI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

            if EHF > 0:
                hotwave_load_mean.append(load[24 * j : 24 * (j + 1)])
                hotwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])

            if ECF == 0 and EHF == 0:
                common_load_mean.append(load[24 * j : 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])


        hot_common_ratio = np.mean(coldwave_load_mean)/np.mean(common_load_mean)-1
        color = cmap(norm(hot_common_ratio))

        print(i)
        target = world[world['NAME'].str.contains(country, case=False)]
        #target = world[world['NAME'].str.contains(country, case=False)]
        #print(target)

        target.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.65, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(coldwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')

    ax_inset_2 = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.15, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    bins = np.linspace(
        0,
        1,
        100  # 分箱数量
    )

    # 计算两个分布的直方图数据(np.mean(np.array(hotwave_tem_norm), axis=1)
    counts_hotwave, bin_edges = np.histogram(np.mean(np.array(coldwave_tem_norm), axis=1), bins=bins)
    counts_common, _ = np.histogram(np.mean(np.array(common_tem_norm), axis=1), bins=bin_edges)  # 使用相同的bin_edges

    ax_inset_2.bar(
        x=bin_edges[:-1],
        height=counts_common,
        width=np.diff(bin_edges),
        # bottom=counts_hotwave,  # 关键参数：堆叠在common的柱子上方
        align='edge',
        color='grey',
        alpha=0.7,
        linewidth=1,
        label='Hotwave Scenario'
    )

    ax_inset_2.bar(
        x=bin_edges[:-1],  # 分箱左边界
        height=counts_hotwave,
        width=np.diff(bin_edges),  # 分箱宽度
        align='edge',  # 柱子对齐分箱边缘
        color='salmon',
        alpha=1,
        linewidth=1,
        label='Common Scenario'
    )

    ax_inset_2.set_xlabel('Norm. Tem.', fontsize=labelsize)
    ax_inset_2.set_ylabel('Density', fontsize=labelsize)
    ax_inset_2.set_yticks([])
    ax_inset_2.set_title('Sample Distribution', fontsize=labelsize)
    ax_inset_2.xaxis.set_tick_params(labelsize=ticksize)

    ax.set_xticks([-10, 0, 10, 20])
    ax.set_yticks([35, 45, 55])
    ax.set_xticklabels(['-10°W', '0°W', '10°W', '20°W'])
    ax.set_yticklabels(['35°N', '45°N', '55°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Patterns', fontsize=labelsize)
    ax_inset.set_xlabel('Hour', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_xlim(-10, 25)
    ax.set_ylim(35, 60)
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(0.8)

    # 设置标题和显示
    ax.set_title("Coldwave in Europe", fontsize=titlesize)
    plt.tight_layout()
    #plt.axis('equal')
    fig.savefig('figures/load_change_in_{}.pdf'.format(region))
    plt.show()


def plot_texas_map(titlesize=16, ticksize=10, labelsize=10, region='texas'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path



    city_name = ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C',
                     'SOUTH_C', 'SOUTHERN', 'WEST']

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)
    usa_states = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip")
    # 筛选出德州数据



    cmap = plt.get_cmap('YlOrRd')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)

    texas = usa_states[usa_states['name'] == 'Texas'].to_crs(world.crs)
    texas.plot(ax=ax, color=cmap(norm(1.2)), edgecolor='white', alpha=0.7, linewidth=1)


    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
    hotwave_tem_norm = []
    common_tem_norm = []
    for i in range(len(city_name)):
        data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(city_name[i]))

        strat_time = '2019/01/01/00'
        end_time = '2023/05/31/23'
        start_date = pd.to_datetime(strat_time)  ## Thursday
        end_date = pd.to_datetime(end_time)

        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]
        data['Data_Hour'] = pd.to_datetime(data['Data_Hour'])  # 确保 Data_Hour 列为 datetime 类型

        load = np.array(data['Load'])
        temperature = np.array(data['Temperature'])

        norm_load = (load - min(load)) / (max(load) - min(load))
        norm_tem = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])
        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)

        # load and temperature slices formulation
        coldwave_index = []
        hotwave_index = []
        common_load_mean = []
        hotwave_load_mean = []
        coldwave_load_mean = []

        for j in range(30, load.shape[0] // 24 - 3 - 30 - 6):

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[j:j + 3]) - T_05
            ECI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                coldwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                coldwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[j:j + 3]) - T_95
            EHI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

            if EHF > 0:
                hotwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                hotwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                hotwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])
            if ECF == 0 and EHF == 0:
                common_load_mean.append(load[24 * j: 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

        hot_common_ratio = np.mean(hotwave_load_mean)/np.mean(common_load_mean)-1
        color = cmap(norm(hot_common_ratio))

        print(i)
        #guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        #guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.65, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(hotwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')

    ax_inset_2 = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.15, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    bins = np.linspace(
        0,
        1,
        100  # 分箱数量
    )

    # 计算两个分布的直方图数据(np.mean(np.array(hotwave_tem_norm), axis=1)
    counts_hotwave, bin_edges = np.histogram(np.mean(np.array(hotwave_tem_norm), axis=1), bins=bins)
    counts_common, _ = np.histogram(np.mean(np.array(common_tem_norm), axis=1), bins=bin_edges)  # 使用相同的bin_edges

    ax_inset_2.bar(
        x=bin_edges[:-1],
        height=counts_common,
        width=np.diff(bin_edges),
        # bottom=counts_hotwave,  # 关键参数：堆叠在common的柱子上方
        align='edge',
        color='grey',
        alpha=0.7,
        linewidth=1,
        label='Hotwave Scenario'
    )

    ax_inset_2.bar(
        x=bin_edges[:-1],  # 分箱左边界
        height=counts_hotwave,
        width=np.diff(bin_edges),  # 分箱宽度
        align='edge',  # 柱子对齐分箱边缘
        color='salmon',
        alpha=1,
        linewidth=1,
        label='Common Scenario'
    )

    ax_inset_2.set_xlabel('Norm. Tem.', fontsize=labelsize)
    ax_inset_2.set_ylabel('Density', fontsize=labelsize)
    ax_inset_2.set_yticks([])
    ax_inset_2.set_title('Sample Distribution', fontsize=labelsize)

    ax.set_yticks([25, 29, 33, 37])
    ax.set_xticks([-105, -100, -95])
    ax.set_xticklabels(['-105°E', '-100°E', '-95°E'])
    ax.set_yticklabels(['25°N', '29°N', '33°N', '37°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Patterns', fontsize=labelsize)
    ax_inset.set_xlabel('Hour', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_xlim(-108, -93)  # 德州经度范围
    ax.set_ylim(22, 38)  # 德州纬度范围
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(0.8)

    # 设置标题和显示
    ax.set_title("Heatwave in Texas", fontsize=titlesize)
    plt.tight_layout()
    #plt.axis('equal')
    fig.savefig('figures/load_change_in_{}.pdf'.format(region))
    plt.show()


def plot_India_map(titlesize=16, ticksize=14, labelsize=14, region='india'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path



    city_name = ['Maharashtra_data_2017_2023',
                    'Delhi_data_2017_2023']

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)
    usa_states = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip")
    # 筛选出德州数据



    cmap = plt.get_cmap('Blues')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)

    india = world[world['NAME'] == 'India']  # 确保为WGS84坐标系
    india.plot(ax=ax, color=cmap(norm(1.2)), edgecolor='white', alpha=0.7, linewidth=1)


    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
    hotwave_tem_norm = []
    common_tem_norm = []
    for i in range(len(city_name)):
        data = pd.read_excel('../Data/reformed_data_updated/India_data_reformed/{}.xlsx'.format(city_name[i]))

        strat_time = '2019/01/01/00'
        end_time = '2021/12/31/23'
        start_date = pd.to_datetime(strat_time)  ## Thursday
        end_date = pd.to_datetime(end_time)

        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]
        data['Data_Hour'] = pd.to_datetime(data['Data_Hour'])  # 确保 Data_Hour 列为 datetime 类型

        load = np.array(data['Load'])
        temperature = np.array(data['Temperature'])

        norm_load = (load - min(load)) / (max(load) - min(load))
        norm_tem = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])
        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)

        # load and temperature slices formulation
        coldwave_index = []
        hotwave_index = []
        common_load_mean = []
        hotwave_load_mean = []
        coldwave_load_mean = []

        for j in range(30, load.shape[0] // 24 - 3 - 30 - 6):

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[j:j + 3]) - T_05
            ECI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                coldwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                coldwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[j:j + 3]) - T_95
            EHI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

            if EHF > 0:
                hotwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                hotwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                hotwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

            if ECF == 0 and EHF == 0:
                common_load_mean.append(load[24 * j: 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])


        hot_common_ratio = np.mean(hotwave_load_mean)/np.mean(common_load_mean)-1
        color = cmap(norm(hot_common_ratio))

        print(i)
        #guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        #guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.65, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(hotwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')

    ax_inset_2 = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.15, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    bins = np.linspace(
        0,
        1,
        100  # 分箱数量
    )

    # 计算两个分布的直方图数据(np.mean(np.array(hotwave_tem_norm), axis=1)
    counts_hotwave, bin_edges = np.histogram(np.mean(np.array(hotwave_tem_norm), axis=1), bins=bins)
    counts_common, _ = np.histogram(np.mean(np.array(common_tem_norm), axis=1), bins=bin_edges)  # 使用相同的bin_edges

    ax_inset_2.bar(
        x=bin_edges[:-1],
        height=counts_common,
        width=np.diff(bin_edges),
        # bottom=counts_hotwave,  # 关键参数：堆叠在common的柱子上方
        align='edge',
        color='grey',
        alpha=0.7,
        linewidth=1,
        label='Hotwave Scenario'
    )

    ax_inset_2.bar(
        x=bin_edges[:-1],  # 分箱左边界
        height=counts_hotwave,
        width=np.diff(bin_edges),  # 分箱宽度
        align='edge',  # 柱子对齐分箱边缘
        color='salmon',
        alpha=1,
        linewidth=1,
        label='Common Scenario'
    )

    ax_inset_2.set_xlabel('Norm. Tem.', fontsize=labelsize)
    ax_inset_2.set_ylabel('Density', fontsize=labelsize)
    ax_inset_2.set_yticks([])
    ax_inset_2.set_title('Sample Distribution', fontsize=labelsize)

    ax.set_yticks([25, 29, 33, 37])
    ax.set_xticks([-105, -100, -95])
    ax.set_xticklabels(['-105°E', '-100°E', '-95°E'])
    ax.set_yticklabels(['25°N', '29°N', '33°N', '37°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Patterns', fontsize=labelsize)
    ax_inset.set_xlabel('Hour', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_xlim(68, 97)  # 德州经度范围
    ax.set_ylim(8, 37)  # 德州纬度范围
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(0.8)

    # 设置标题和显示
    ax.set_title("Heatwave in India", fontsize=titlesize)
    plt.tight_layout()
    #plt.axis('equal')
    fig.savefig('figures/load_change_in_{}.pdf'.format(region))
    plt.show()


def plot_PJM_map(titlesize=16, ticksize=14, labelsize=14, region='pjm'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path



    city_name = ['Allegheny Power System',
                'American Electric Power Co., Inc', 'Atlantic Electric Company',
                'Baltimore Gas and Electric Company',
                'ComEd', 'Dayton Power and Light Company', 'Delmarva Power and Light',
                'Dominion Energy', 'Duke Energy Ohio', 'Duquesne Light',
                'East Kentucky Power Coop', 'First Energy - Pennsylvania Electric Company',
                'Jersey Central Power and Light Company', 'Metropolitan Edison Company',
                'Orion Energy', 'Pennsylvania Electric Company',
                'Pennsylvania Power and Light Company',
                'Potomac Electric Power',
                'Public Service Electric and Gas Company']

    pjm_states = [
        'Pennsylvania', 'New Jersey', 'Maryland', 'Delaware',
        'Virginia', 'West Virginia', 'Ohio', 'Kentucky',
        'Illinois', 'Indiana', 'Michigan', 'North Carolina', 'Tennessee'
    ]


    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)
    states = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip")
    pjm = states[states['name'].isin(pjm_states)]  # 转为Web墨卡托投影
    # 筛选出德州数据



    cmap = plt.get_cmap('Greens')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)





    #india = world[world['NAME'] == 'United States']  # 确保为WGS84坐标系
    #india.plot(ax=ax, color=cmap(norm(1.2)), edgecolor='white', alpha=0.7, linewidth=1)
    color_values = np.linspace(0.15, 0.6, len(pjm_states))
    for i, state in enumerate(pjm_states):
        pjm[pjm['name'] == state].plot(
            ax=ax,
            color=cmap(norm(color_values[i])),  # 使用递增的颜色值
            edgecolor='white',
            linewidth=0.8,
            label=state
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
    hotwave_tem_norm = []
    common_tem_norm = []
    for i in range(len(city_name)):
        data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(city_name[i]), header=0,
                           usecols=['Date_Hour', 'Load', 'Temperature'])

        strat_time = '2021/03/28/00'
        end_time = '2023/05/31/23'
        start_date = pd.to_datetime(strat_time)  ## Thursday
        end_date = pd.to_datetime(end_time)



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

        load = np.array(data['Load'])
        temperature = np.array(data['Temperature'])

        norm_load = (load - min(load)) / (max(load) - min(load))
        norm_tem = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])
        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)

        # load and temperature slices formulation
        coldwave_index = []
        hotwave_index = []
        common_load_mean = []
        hotwave_load_mean = []
        coldwave_load_mean = []

        for j in range(30, load.shape[0] // 24 - 3 - 30 - 6):

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[j:j + 3]) - T_05
            ECI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                coldwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                coldwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[j:j + 3]) - T_95
            EHI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

            if EHF > 0:
                hotwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                hotwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                hotwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

            if ECF == 0 and EHF == 0:
                common_load_mean.append(load[24 * j: 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

        hot_common_ratio = np.mean(hotwave_load_mean)/np.mean(common_load_mean)-1
        color = cmap(norm(hot_common_ratio))

        print(i)
        #guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        #guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.65, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(hotwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')

    ax_inset_2 = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.15, 0.15, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    bins = np.linspace(
        0,
        1,
        100  # 分箱数量
    )

    # 计算两个分布的直方图数据(np.mean(np.array(hotwave_tem_norm), axis=1)
    counts_hotwave, bin_edges = np.histogram(np.mean(np.array(hotwave_tem_norm), axis=1), bins=bins)
    counts_common, _ = np.histogram(np.mean(np.array(common_tem_norm), axis=1), bins=bin_edges)  # 使用相同的bin_edges

    ax_inset_2.bar(
        x=bin_edges[:-1],
        height=counts_common,
        width=np.diff(bin_edges),
        # bottom=counts_hotwave,  # 关键参数：堆叠在common的柱子上方
        align='edge',
        color='grey',
        alpha=0.7,
        linewidth=1,
        label='Hotwave Scenario'
    )

    ax_inset_2.bar(
        x=bin_edges[:-1],  # 分箱左边界
        height=counts_hotwave,
        width=np.diff(bin_edges),  # 分箱宽度
        align='edge',  # 柱子对齐分箱边缘
        color='salmon',
        alpha=1,
        linewidth=1,
        label='Common Scenario'
    )

    ax_inset_2.set_xlabel('Norm. Tem.', fontsize=labelsize)
    ax_inset_2.set_ylabel('Density', fontsize=labelsize)
    ax_inset_2.set_yticks([])
    ax_inset_2.set_title('Sample Distribution', fontsize=labelsize)

    ax.set_xlim(-93, -66)  # 经度范围（西→东）
    ax.set_ylim(28, 45)  # 纬度范围（南→北）
    ax.set_yticks([30, 35, 40, 45])
    ax.set_xticks([-90, -80, -70])
    ax.set_xticklabels(['-90°E', '-80°E', '-70°E'])
    ax.set_yticklabels(['30°N', '35°N', '40°N', '45°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Patterns', fontsize=labelsize)
    ax_inset.set_xlabel('Hour', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(0.8)

    # 设置标题和显示
    ax.set_title("Heatwave in PJM", fontsize=titlesize)
    plt.tight_layout()
    fig.savefig('figures/load_change_in_{}.pdf'.format(region))
    #plt.axis('equal')
    plt.show()


#plot_guangdong_map()

#plot_hunan_map()

#plot_europe_map()

#plot_texas_map()

#plot_India_map()

#plot_PJM_map()

def plot_distribution_original(titlesize=16, ticksize=14, labelsize=14):
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
    # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))

    #ax = ax.flatten()

    k = 0
    for country in ['Sweden']:

        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2015/01/01/00')
        end_date = pd.to_datetime('2017/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

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
        for i in range(30, load.shape[0] // 24 - 3 - 30 - 6):
            ## load and temperature
            load_slice_list.append(load[24 * i:24 * (i + 8)])
            tem_slice_list.append(temperature[24 * i:24 * (i + 8)])

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

        coldwave_samples = generate_coldwave_samples(country, num_samples=400, weather_type='coldwave').cpu().detach().numpy()
        print(coldwave_samples[0, 0])

        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)
        print(coldwave_samples[0, 0])

        load_slice_list = np.array(load_slice_list)
        #print(load_slice_list.shape)
        tem_slice_list = np.array(tem_slice_list)

        sample_load_list = np.mean(load_slice_list[:, :-24], axis=1)
        label_load_list = np.mean(load_slice_list[:, -24:], axis=1)
        sample_tem_list = np.mean(tem_slice_list[:, :-24], axis=1)
        label_tem_list = np.mean(tem_slice_list[:, -24:], axis=1)
        generated_sample_load_list = np.mean(coldwave_samples[:, 0, :-24], axis=1)
        generated_sample_tem_list = np.mean(coldwave_samples[:, 1, :-24], axis=1)

        # 设置统一的bins（确保两个直方图使用相同分箱）
        bins = np.linspace(
            0,
            1,
            50  # 分箱数量
        )

        # 绘制概率密度直方图（关键参数：density=True）
        ax.hist(sample_tem_list, bins=bins, density=True,
                 alpha=0.4, label='Real Samples', color='crimson')
        ax.hist(generated_sample_tem_list, bins=bins, density=True,
                 alpha=0.4, label='Generated Samples', color='cadetblue')

        # 添加核密度曲线（KDE）增强可视化
        kde_real = gaussian_kde(sample_tem_list)
        kde_gen = gaussian_kde(generated_sample_tem_list)
        x_vals = np.linspace(bins[0], bins[-1], 500)
        ax.plot(x_vals, kde_real(x_vals), color='red', linewidth=2, linestyle='--')
        ax.plot(x_vals, kde_gen(x_vals), color='darkgreen', linewidth=2, linestyle='--')


        # 添加标注
        ax.set_xlabel('Norm. Temperature', fontsize=labelsize)
        ax.set_ylabel('Density', fontsize=labelsize)  # 纵轴标签改为概率密度
        ax.xaxis.set_tick_params(labelsize=ticksize)
        ax.yaxis.set_tick_params(labelsize=ticksize)
        ax.set_xlim(0, 1)
        ax.set_xticks([0.5, 1])
        #.set_ylim(0, 5)
        #ax.set_title(country, fontsize=titlesize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        k += 1
        #plt.legend()
        #plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

#plot_distribution_original()





def ablation_figure_multi(sheet_name_list, title_list, metric='MAE'):
    font_size = 14
    import matplotlib
    matplotlib.rcParams.update({
        'font.size': font_size,  # 全局字体大小
        'axes.titlesize': font_size + 2,  # 子图标题
        'axes.labelsize': font_size,  # 坐标轴标题
        'xtick.labelsize': font_size,  # x轴刻度
        'ytick.labelsize': font_size,  # y轴刻度
        'legend.fontsize': font_size,  # 图例     # 图表标题
    })
    Model = ['MLP', 'LSTM', 'CNN']
    Setting = ['+Separation', '', '+DA']
    label_list = ['Backbone', 'Backbone+ES', 'Backbone+ESDF']
    color_list = ["#A8CDECFF", "#F6955EFF", "#682C37FF"]

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    axes = axes.flatten()

    for idx, sheet_name in enumerate(sheet_name_list):
        data = pd.read_excel('../极端温度实验结果.xlsx', sheet_name=sheet_name)
        cols = ['Country']
        combined = []
        for model in Model:
            for setting in Setting:
                if setting == '':
                    combined.append(model)
                else:
                    combined.append(f"{model}{setting}")
        combined = [f"{i}_{metric}" for i in combined for metric in ['MAE', 'RMSE']]
        print(combined)
        cols = cols + combined
        data.columns = cols
        data = data.iloc[1:].reset_index(drop=True)
        MAE_col = [col for col in data.columns if 'MAE' in col]
        var_MAE_value = np.var(data.iloc[:-1][MAE_col], axis=0).to_numpy()
        MAE_col = ['MLP_MAE', 'MLP+DA_MAE', 'MLP+Separation_MAE',
                   'LSTM_MAE', 'LSTM+DA_MAE', 'LSTM+Separation_MAE',
                   'CNN_MAE', 'CNN+DA_MAE', 'CNN+Separation_MAE']

        RMSE_col = ['MLP_RMSE', 'MLP+DA_RMSE', 'MLP+Separation_RMSE',
                    'LSTM_RMSE', 'LSTM+DA_RMSE', 'LSTM+Separation_RMSE',
                    'CNN_RMSE', 'CNN+DA_RMSE', 'CNN+Separation_RMSE']

        # MAE_col=[col for col in data.columns if 'MAE' in col]
        # RMSE_col = [col for col in data.columns if 'RMSE' in col]

        col_for_draw = MAE_col if metric == 'MAE' else RMSE_col

        min_value = np.min(data.iloc[:-1][col_for_draw], axis=0).to_numpy().reshape(3, 3)
        max_value = np.max(data.iloc[:-1][col_for_draw], axis=0).to_numpy().reshape(3, 3)

        # var_MAE_value = np.var(data.iloc[:-1][MAE_col], axis=0).to_numpy()#.reshape(3, 3)
        mean_value = np.mean(data.iloc[:-1][col_for_draw], axis=0).to_numpy()  # .reshape(3, 3)
        # min_MAE_value = mean_MAE_value - np.sqrt(list(var_MAE_value))
        # max_MAE_value = mean_MAE_value + np.sqrt(list(var_MAE_value))

        mean_value = mean_value.reshape(3, 3)
        # min_MAE_value= min_MAE_value.reshape(3, 3)
        # max_MAE_value= max_MAE_value.reshape(3, 3)

        x = np.arange(len(Model))
        bar_width = 0.3  # 减小柱子宽度
        gap = 0.08  # 新增：组内柱子间隔

        ax = axes[idx]
        for i in range(len(Setting)):
            # 调整x坐标：增加间隔 (gap)
            if i == 2:
                continue


            x_pos = x + i * (bar_width + gap)  # 关键修改
            y = mean_value[:, i]
            ax.bar(
                x_pos,
                y,
                width=bar_width,
                capsize=5,
                label=label_list[i],
                color=color_list[i],
                edgecolor='black',
                linewidth=1
            )



        ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0, lw=1.5)  # 添加横向网格线，zorder=0保证在bar下方
        for patch in ax.patches:
            patch.set_zorder(2)  # 柱子zorder调高，确保在网格线上方
            # yerr = np.vstack([
            #     y - min_value[:, i],
            #     max_value[:, i] - y
            # ])
            # ax.bar(x + i * bar_width, y, width=bar_width, yerr=yerr, capsize=5, label=label_list[i], color=color_list[i])
        # ax.bar(x + i * bar_width, y, width=bar_width, capsize=5, label=label_list[i], color=color_list[i])
        center = x + bar_width * (len(Setting) - 1) / 2
        ax.set_xticks(center)
        ax.set_xticklabels(Model)
        ax.set_title(title_list[idx])
        ax.set_ylim(0, 1.4 * np.max(mean_value))
        #if idx % 3 == 0:
        #    ax.set_ylabel(metric)
        ax.set_ylabel(metric)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    # 合并legend到最下方
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,  # 图例列数（根据条目数量调整）
            frameon=True,  # 是否显示边框
            fontsize=font_size,
            edgecolor='black')
    #plt.savefig('../figures/ablation_figure_{}.png'.format(metric), bbox_inches='tight', dpi=300)
    plt.show()

def ablation_figure_multi_3d(sheet_name_list, title_list, metric='MAE',
                             titlesize=16, ticksize=14, labelsize=14):
    font_size = 14
    import matplotlib
    matplotlib.rcParams.update({
        'ytick.labelsize': font_size,  # y轴刻度
    })
    Model = ['MLP', 'LSTM', 'CNN']
    Setting = ['+Separation', '', '+DA']
    label_list = ['Backbone', 'Backbone+DA', 'Backbone+Separation']
    color_list = ['palevioletred', 'lavenderblush']

    fig, axes = plt.subplots(1, 6, figsize=(18, 4.5))
    axes = axes.flatten()

    mae_proposed = []
    mae_basic = []
    rmse_proposed = []
    rmse_basic = []


    for idx, sheet_name in enumerate(sheet_name_list):
        data = pd.read_excel('../极端温度实验结果.xlsx', sheet_name=sheet_name)
        cols = ['Country']
        combined = []
        for model in Model:
            for setting in Setting:
                if setting == '':
                    combined.append(model)
                else:
                    combined.append(f"{model}{setting}")
        combined = [f"{i}_{metric}" for i in combined for metric in ['MAE', 'RMSE']]
        print(combined)
        cols = cols + combined
        data.columns = cols
        data = data.iloc[1:].reset_index(drop=True)
        MAE_col = [col for col in data.columns if 'MAE' in col]

        MAE_col = ['LSTM_MAE', 'LSTM+DA_MAE', 'LSTM+Separation_MAE']

        RMSE_col = ['LSTM_RMSE', 'LSTM+DA_RMSE', 'LSTM+Separation_RMSE']

        # MAE_col=[col for col in data.columns if 'MAE' in col]
        # RMSE_col = [col for col in data.columns if 'RMSE' in col]

        col_for_draw = MAE_col if metric == 'MAE' else RMSE_col

        mean_value = np.mean(data.iloc[:-1][col_for_draw], axis=0).to_numpy()  # .reshape(3, 3)

        mean_value_mae = np.mean(data.iloc[:-1][MAE_col], axis=0).to_numpy()  # .reshape(3, 3)
        mean_value_rmse = np.mean(data.iloc[:-1][RMSE_col], axis=0).to_numpy()  # .reshape(3, 3)

        mean_value = mean_value.reshape(1, 3)

        mean_value_mae = mean_value_mae.reshape(1, 3)
        mean_value_rmse = mean_value_rmse.reshape(1, 3)


        x = np.arange(1)
        bar_width = 0.3  # 减小柱子宽度
        gap = 0.08  # 新增：组内柱子间隔

        ax = axes[idx]
        for i in range(len(Setting)):
            # 调整x坐标：增加间隔 (gap)
            if i == 2:
                continue


            x_pos = x + i * (bar_width + gap)  # 关键修改
            y = mean_value_mae[:, i]
            ax.bar(
                x_pos,
                y,
                width=bar_width,
                capsize=5,
                label=label_list[i],
                color=color_list[i],
                edgecolor='black',
                linewidth=1
            )

            x_pos = x+1 + i * (bar_width + gap)  # 关键修改
            y = mean_value_rmse[:, i]
            ax.bar(
                x_pos,
                y,
                width=bar_width,
                capsize=5,
                color=color_list[i],
                edgecolor='black',
                linewidth=1
            )



        ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0, lw=1.5)  # 添加横向网格线，zorder=0保证在bar下方
        for patch in ax.patches:
            patch.set_zorder(2)  # 柱子zorder调高，确保在网格线上方



        center = [x + (bar_width+gap)/2 for x in range(2)]
        ax.set_xticks(center)
        ax.set_yticks([0, 0.02, 0.04, 0.06])
        ax.set_xticklabels(['nMAE', 'nRMSE'], fontsize=14)
        ax.set_title(title_list[idx], fontsize=titlesize)
        ax.set_ylim(0, 0.07)
        #if idx % 3 == 0:
        #    ax.set_ylabel(metric)
        ax.set_ylabel('Forecasting Error', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(fontsize=10)
    #plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.tight_layout(rect=[0, 0, 1, 0.85], w_pad=4)

    fig.suptitle("Forecasting Results with/without Generated Samples",
                 y=1, fontsize=titlesize)

    # 合并legend到最下方
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center', ncol=3,  # 图例列数（根据条目数量调整）
    #        frameon=True,  # 是否显示边框
    #        fontsize=font_size,
    #        edgecolor='black')

    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.95),  # 调整垂直位置
               ncol=4,  # 设置列数以水平排列
               fontsize=font_size,
               frameon=False)
    #plt.savefig('../figures/ablation_figure_{}.png'.format(metric), bbox_inches='tight', dpi=300)
    plt.savefig('figures/dataset_balance_performance.pdf')
    plt.show()


# 用法示例
sheet_name_list = [
    '嵌入（欧洲2018寒潮）',
    '嵌入（德州2023热浪）',
    '嵌入（PJM2023热浪）',
    '嵌入（印度2022热浪）',
    '嵌入（广东2023热浪）',
    '嵌入（湖南2023寒潮）'
]

title_list = [
    'Europe',
    'Texas',
    'PJM',
    'India',
    'Guangdong',
    'Hunan'
]

#ablation_figure_multi(sheet_name_list, title_list)

#ablation_figure_multi_3d(sheet_name_list, title_list)



def plot_sample_coldwave(lw=1, lw2=1.0, generation=True, index=0):
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
    #from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe
    for country in ['France']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2015/01/01/00')
        end_date = pd.to_datetime('2017/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)

        # load and temperature slices formulation
        load_slice_list = []
        tem_slice_list = []
        extreme_load_list = []
        extreme_tem_list = []
        weekday_index_list = []
        coldwave_index = []
        hotwave_index = []
        for i in range(30, load.shape[0]//192 - 3):
            ## load and temperature
            load_slice_list.append(load[192 * i:192 * i + 192])
            tem_slice_list.append(temperature[192 * i:192 * i + 192])

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[8*i:8*i + 3]) - T_05
            ECI_accl = np.mean(T_i_list[8*i:8*i + 3]) - np.mean(T_i_list[i - 30:i])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                extreme_load_list.append(load[192 * i:192 * i + 192])
                extreme_tem_list.append(temperature[192 * i:192 * i + 192])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[i:i + 3]) - T_95
            EHI_accl = np.mean(T_i_list[i:i + 3]) - np.mean(T_i_list[i - 30:i])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))


        coldwave_samples = generate_coldwave_samples(country, num_samples=50).cpu().detach().numpy()
        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)

        load_slice_list = np.array(load_slice_list)
        print(load_slice_list.shape)
        tem_slice_list = np.array(tem_slice_list)

        print(load_slice_list)


        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.5), dpi=900)
        # 计算均值、最大值和最小值
        mean_load = np.mean(load_slice_list, axis=0)
        max_load = np.max(load_slice_list, axis=0)
        min_load = np.min(load_slice_list, axis=0)

        mean_tem = np.mean(tem_slice_list, axis=0)
        max_tem = np.max(tem_slice_list, axis=0)
        min_tem = np.min(tem_slice_list, axis=0)
        x = np.arange(192)

        #print(mean_load)

        # 绘制均值曲线
        #ax[0].plot(x, mean_curve, color='blue', label='Mean Curve')

        # 绘制包络线
        #line1 = ax[0].fill_between(x, min_load, max_load, color='gray', alpha=0.2, label='Original Samples')


        #ax[0].plot(x, coldwave_samples[1, 0, :], color='blue')





        #ax[1].fill_between(x, min_tem, max_tem, color='gray', alpha=0.2, label='Envelope')


        if generation:
            ax.plot(x, coldwave_samples[index, 0, :], color='darkred',
                    lw=lw2)


        else:
            ax.plot(x, load_slice_list[index, :], color='skyblue',
                    lw=lw)



        # 在图像底部添加共同图例
        #fig.legend(
        #    handles=handles,
        #    labels=labels,
        #    loc="lower center",  # 图例位置（底部居中）
        #    bbox_to_anchor=(0.5, 0),  # 调整位置（x, y偏移）
        #    ncol=1,  # 图例列数（根据条目数量调整）
        #    frameon=True,  # 是否显示边框
        #    fontsize=12,
        #    edgecolor='black'
        #)



        #ax[0].set_ylabel('Norm. Load', fontsize=14)
        #ax[1].set_ylabel('Norm. Tem.', fontsize=14)
        #ax[1].set_xlabel('Time Index', fontsize=14)

        ax.tick_params(axis='both', labelsize=12)
        #ax[1].tick_params(axis='both', labelsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax[1].spines['top'].set_visible(False)
        #ax[1].spines['right'].set_visible(False)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15, top=0.85)  # 预留底部空间
        #ax[0].set_title('Sample Visualization', fontsize=16)
        plt.show()


#plot_sample_coldwave(generation=False, index=25)


def plot_ratio_generation(titlesize=12, ticksize=12, labelsize=12):

    ratio_list = ['0', '0.1', '0.25', '0.5', '0.75', '1']
    region_list = ['欧洲', '广东', '德州', 'PJM', '湖南', '印度']
    region_name = ['GD', 'PJM', 'Texas', 'India', 'Europe', 'HN']
    model_list = ['MLP', 'LSTM', 'CNN']


    result_list = []
    result_list_rmse = []

    for i in range(len(ratio_list)):


        region_result = []
        region_result_rmse = []
        for j in range(len(region_list)):
            data = pd.read_excel('generation_validation/generation_data_{}.xlsx'.format(ratio_list[i]),
                                 sheet_name=region_list[j])

            data = data.iloc[1:].reset_index(drop=True).to_numpy()
            #print(data[-1, 1])

            region_result.append([data[-1, 1], data[-1, 3], data[-1, 5]])
            region_result_rmse.append([data[-1, 2], data[-1, 4], data[-1, 6]])

        result_list.append(region_result)
        result_list_rmse.append(region_result_rmse)


    result_list = np.array(result_list)
    result_list_rmse = np.array(result_list_rmse)


    print(result_list.shape)

    color1 = '#699ECA'
    color2 = '#F16E65'
    color3 = '#008A89'

    fig = plt.figure(figsize=(16, 4.5))

    # 定义具体的宽度比例（可以根据需要调整）
    gs = GridSpec(1, 4, width_ratios=[1.5, 2, 1.5, 2])  # 3:2:3:2的比例



    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])


    ## ax1的散点图
    colors = ['#FFB6C1', '#FF69B4', '#FF1493', '#DC143C', '#1976D2']  # 粉色到红色渐变
    # 或者使用其他颜色方案：
    # colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2']  # 蓝色渐变
    # colors = ['#E8F5E8', '#81C784', '#4CAF50', '#2E7D32']  # 绿色渐变

    # 获取 ratio=0 时的精度作为 x 轴数据
    x_data = result_list[0, :, :].flatten()  # ratio=0 的所有数据

    # 为每个非零 ratio 创建散点
    scatters = []
    for i, ratio in enumerate(['0.1', '0.25', '0.5', '0.75', '1']):
        ratio_idx = i + 1  # 在 ratio_list 中的索引
        y_data = result_list[ratio_idx, :, :].flatten()  # 当前 ratio 的所有数据

        # 绘制散点
        scatter = ax1.scatter(x_data, y_data,
                              c=colors[i],
                              alpha=0.7,
                              s=60,
                              label=f'ratio={ratio}',
                              edgecolors='black',
                              linewidth=0.5)
        scatters.append(scatter)

    ax1.set_xlabel('nMAE without Synthetic Samples', fontsize=labelsize)
    ax1.set_ylabel('nMAE with Synthetic Samples', fontsize=labelsize)

    # 添加对角线 (y=x) 作为参考
    min_val = min(result_list.min(), result_list.min())
    max_val = max(result_list.max(), result_list.max())
    ax1.plot([min_val, max_val], [min_val, max_val],
             'k--', alpha=0.5, linewidth=1, label='y=x')

    ax2.set_xlabel('nMAE without Synthetic Samples', fontsize=labelsize)
    ax2.set_ylabel('nMAE with different Synthetic Sample Ratios', fontsize=labelsize)



    ## ax2的折线图
    avg_MLP = np.mean(result_list[:, :, 0], axis=1)
    max_MLP = np.max(result_list[:, :, 0], axis=1)
    min_MLP = np.min(result_list[:, :, 0], axis=1)

    avg_LSTM = np.mean(result_list[:, :, 1], axis=1)
    max_LSTM = np.max(result_list[:, :, 1], axis=1)
    min_LSTM = np.min(result_list[:, :, 1], axis=1)

    avg_CNN = np.mean(result_list[:, :, 2], axis=1)
    max_CNN = np.max(result_list[:, :, 2], axis=1)
    min_CNN = np.min(result_list[:, :, 2], axis=1)

    x = [0, 0.1, 0.25, 0.5, 0.75, 1]

    line1, = ax2.plot(x, avg_MLP, marker='o', markersize=15, lw=4,
             markerfacecolor='white', markeredgewidth=3, color=color1,
                      label='MLP')
    ax2.fill_between(x, min_MLP, max_MLP, color=color1, alpha=0.2)

    line2, = ax2.plot(x, avg_LSTM, marker='*', markersize=15, lw=4,
             markerfacecolor='white', markeredgewidth=3, color=color2,
                      label='LSTM')
    ax2.fill_between(x, min_LSTM, max_LSTM, color=color2, alpha=0.2)

    line3, = ax2.plot(x, avg_CNN, marker='h', markersize=15, lw=4,
             markerfacecolor='white', markeredgewidth=3, color=color3,
                      label='CNN')
    ax2.fill_between(x, min_CNN, max_CNN, color=color3, alpha=0.2)



    handles = scatters + [line1, line2, line3]  # 所有线条的句柄
    labels = [h.get_label() for h in handles]  # 对应的标签

    ax2.set_xlabel('Synthetic Samples Penetration Ratio', fontsize=labelsize)
    ax2.set_ylabel('nRMSE', fontsize=labelsize)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)



    # ax3的散点图
    x_data = result_list_rmse[0, :, :].flatten()  # ratio=0 的所有数据

    # 为每个非零 ratio 创建散点
    for i, ratio in enumerate(['0.1', '0.25', '0.5', '0.75', '1']):
        ratio_idx = i + 1  # 在 ratio_list 中的索引
        y_data = result_list_rmse[ratio_idx, :, :].flatten()  # 当前 ratio 的所有数据

        # 绘制散点
        scatter = ax3.scatter(x_data, y_data,
                              c=colors[i],
                              alpha=0.7,
                              s=60,
                              label=f'ratio={ratio}',
                              edgecolors='black',
                              linewidth=0.5)


    ax3.set_xlabel('nRMSE without Synthetic Samples', fontsize=labelsize)
    ax3.set_ylabel('nRMSE with Synthetic Samples', fontsize=labelsize)

    # 添加对角线 (y=x) 作为参考
    min_val = min(result_list_rmse.min(), result_list_rmse.min())
    max_val = max(result_list_rmse.max(), result_list_rmse.max())
    ax3.plot([min_val, max_val], [min_val, max_val],
             'k--', alpha=0.5, linewidth=1, label='y=x')

    # ax4的折线图
    avg_MLP = np.mean(result_list_rmse[:, :, 0], axis=1)
    max_MLP = np.max(result_list_rmse[:, :, 0], axis=1)
    min_MLP = np.min(result_list_rmse[:, :, 0], axis=1)

    avg_LSTM = np.mean(result_list_rmse[:, :, 1], axis=1)
    max_LSTM = np.max(result_list_rmse[:, :, 1], axis=1)
    min_LSTM = np.min(result_list_rmse[:, :, 1], axis=1)

    avg_CNN = np.mean(result_list_rmse[:, :, 2], axis=1)
    max_CNN = np.max(result_list_rmse[:, :, 2], axis=1)
    min_CNN = np.min(result_list_rmse[:, :, 2], axis=1)

    x = [0, 0.1, 0.25, 0.5, 0.75, 1]

    line1, = ax4.plot(x, avg_MLP, marker='o', markersize=15, lw=4,
                      markerfacecolor='white', markeredgewidth=3, color=color1,
                      label='MLP')
    ax4.fill_between(x, min_MLP, max_MLP, color=color1, alpha=0.2)

    line2, = ax4.plot(x, avg_LSTM, marker='*', markersize=15, lw=4,
                      markerfacecolor='white', markeredgewidth=3, color=color2,
                      label='LSTM')
    ax4.fill_between(x, min_LSTM, max_LSTM, color=color2, alpha=0.2)

    line3, = ax4.plot(x, avg_CNN, marker='h', markersize=15, lw=4,
                      markerfacecolor='white', markeredgewidth=3, color=color3,
                      label='CNN')
    ax4.fill_between(x, min_CNN, max_CNN, color=color3, alpha=0.2)


    ax4.set_xlabel('Synthetic Samples Penetration Ratio', fontsize=labelsize)
    ax4.set_ylabel('nRMSE', fontsize=labelsize)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)




    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.93),  # 调整垂直位置
               ncol=8,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=False)

    # fig.subplots_adjust(top=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.85])



    #plt.tight_layout()
    plt.show()


#plot_ratio_generation()

def plot_convergence_curve(titlesize=12, ticksize=12, labelsize=12, zoomx1=60, xlim=-0.1, tlim=0.45):

    region_list = ['', '_Guangdong', '_hunan', '_India', '_Texas']
    color1 = '#699ECA'
    color2 = '#F16E65'
    color3 = '#008A89'
    colors = [color1, color2, color3]

    country_lists = [['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland'],

                     ['chaozhou', 'dongguan', 'foshan', 'guangzhou', 'heyuan', 'huizhou',
                      'jiangmen', 'jieyang', 'maoming', 'meizhou', 'qingyuan', 'shantou',
                      'shanwei', 'shaoguan', 'shenzhen', 'yangjiang', 'yunfu', 'zhanjiang',
                      'zhaoqing', 'zhongshan', 'zhuhai'],

                     ['娄底', '岳阳', '常德', '张家界', '怀化',
                      '株洲', '永州', '湘潭', '湘西', '益阳',
                      '衡阳', '邵阳', '郴州', '长沙'],

                     ['Maharashtra_data_2017_2023',
                      'Delhi_data_2017_2023'],

                     ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C',
                      'SOUTH_C', 'SOUTHERN', 'WEST']]

    model_list = ['ANN', 'LSTM', 'CNN']

    convergence_lists_ANN = []
    convergence_lists_LSTM = []
    convergence_lists_CNN = []

    for i in range(len(region_list)):
        convergence_list_ANN = []
        convergence_list_LSTM = []
        convergence_list_CNN = []
        for j in range(len(country_lists[i])):
            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/convergence_{}_ANN.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values[:100]
            convergence_list_ANN.append(data)

            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/convergence_{}_LSTM.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values[:100]
            convergence_list_LSTM.append(data)

            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/convergence_{}_CNN.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values[:100]
            convergence_list_CNN.append(data)

        convergence_lists_ANN.append(convergence_list_ANN)
        convergence_lists_LSTM.append(convergence_list_LSTM)
        convergence_lists_CNN.append(convergence_list_CNN)

    #convergence_lists_ANN = np.array(convergence_lists_ANN)
    #convergence_lists_LSTM = np.array(convergence_lists_LSTM)
    #convergence_lists_CNN = np.array(convergence_lists_CNN)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    flatten_ANN = [item for sublist in convergence_lists_ANN for item in sublist]
    flatten_LSTM = [item for sublist in convergence_lists_LSTM for item in sublist]
    flatten_CNN = [item for sublist in convergence_lists_CNN for item in sublist]

    #print(flatten_ANN)
    results = [flatten_ANN, flatten_LSTM, flatten_CNN]


    x = range(100)
    ax[0].plot(np.mean(np.array(flatten_ANN), axis=0), color=color1, lw=2, label='Backbone: MLP')
    ax[0].fill_between(x,
                    np.max(np.array(flatten_ANN), axis=0),
                    np.min(np.array(flatten_ANN), axis=0),
                    color=color1, alpha=0.2)

    ax[1].plot(np.mean(np.array(flatten_LSTM), axis=0), color=color2, lw=2, label='Backbone: LSTM')
    ax[1].fill_between(x,
                    np.max(np.array(flatten_LSTM), axis=0),
                    np.min(np.array(flatten_LSTM), axis=0),
                    color=color2, alpha=0.2)

    ax[2].plot(np.mean(np.array(flatten_CNN), axis=0), color=color3, lw=2, label='Backbone: CNN')
    ax[2].fill_between(x,
                    np.max(np.array(flatten_CNN), axis=0),
                    np.min(np.array(flatten_CNN), axis=0),
                    color=color3, alpha=0.2)

    for i in range(3):
        ax[i].set_ylabel('Separation Loss', fontsize=labelsize)
        ax[i].set_xlabel('Training Epoch', fontsize=labelsize)
        ax[i].xaxis.set_tick_params(labelsize=ticksize)
        ax[i].yaxis.set_tick_params(labelsize=ticksize)

        ax[i].legend(loc='upper center',
               bbox_to_anchor=(0.25, 1),  # 调整垂直位置
               ncol=1,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=False)


        # zoom1
        zoomx2 = zoomx1 + 10
        axins_1 = ax[i].inset_axes((0.55, 0.4, 0.4, 0.3))
        axins_1.plot(np.mean(np.array(results[i]), axis=0), color=colors[i], lw=2)
        #axins_1.plot(np.mean(np.array(flatten_LSTM), axis=0), color=color2, lw=2)
        #axins_1.plot(np.mean(np.array(flatten_CNN), axis=0), color=color3, lw=2)

        axins_1.set_xlim(zoomx1, zoomx2)
        axins_1.set_ylim(xlim, tlim)
        axins_1.set_title('Zoom in', fontsize=labelsize)
        tx0 = zoomx1
        tx1 = zoomx2
        ty0 = xlim
        ty1 = tlim
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        ax[i].plot(sx, sy, "black")
        xy = (tx0, ty1)
        xy2 = (tx0, ty0)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax[i])
        con.set_color('silver')
        axins_1.add_artist(con)
        xy = (tx1, ty1)
        xy2 = (tx1, ty0)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax[i])
        con.set_color('silver')
        axins_1.add_artist(con)

        axins_1.set_xticks([60, 65, 70])
        axins_1.set_yticks([0, 0.2, 0.4])
        axins_1.xaxis.set_tick_params(labelsize=ticksize)
        axins_1.yaxis.set_tick_params(labelsize=ticksize)

        ax[i].set_title('RD Network Convergence', fontsize=titlesize)


        ax[i].set_ylim(-0.2, 6)
        ax[i].margins(x=0)

    plt.tight_layout()
    plt.savefig('figures/RD_convergence.pdf')
    plt.show()


#plot_convergence_curve()


def plot_heatbar_ablation_error(titlesize=12, ticksize=12, labelsize=12):
    #region_list = ['', '_Guangdong', '_hunan', '_India', '_PJM', '_Texas']
    region_list = ['', '_Guangdong']
    color1 = '#699ECA'
    color2 = '#F16E65'
    color3 = '#008A89'

    country_lists = [['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                      'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland'],

                     ['chaozhou', 'dongguan', 'foshan', 'guangzhou', 'heyuan', 'huizhou',
                      'jiangmen', 'jieyang', 'maoming', 'meizhou', 'qingyuan', 'shantou',
                      'shanwei', 'shaoguan', 'shenzhen', 'yangjiang', 'yunfu', 'zhanjiang',
                      'zhaoqing', 'zhongshan', 'zhuhai'],

                     ['娄底', '岳阳', '常德', '张家界', '怀化',
                      '株洲', '永州', '湘潭', '湘西', '益阳',
                      '衡阳', '邵阳', '郴州', '长沙'],

                     ['Maharashtra_data_2017_2023',
                      'Delhi_data_2017_2023'],

                     ['Allegheny Power System',
                      'American Electric Power Co., Inc', 'Atlantic Electric Company',
                      'Baltimore Gas and Electric Company',
                      'ComEd', 'Dayton Power and Light Company', 'Delmarva Power and Light',
                      'Dominion Energy', 'Duke Energy Ohio', 'Duquesne Light',
                      'East Kentucky Power Coop', 'First Energy - Pennsylvania Electric Company',
                      'Jersey Central Power and Light Company', 'Metropolitan Edison Company',
                      'Orion Energy', 'Pennsylvania Electric Company',
                      'Pennsylvania Power and Light Company',
                      'Potomac Electric Power',
                      'Public Service Electric and Gas Company'],

                     ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C',
                      'SOUTH_C', 'SOUTHERN', 'WEST']]

    model_list = ['ANN', 'LSTM', 'CNN']


    mae_list_proposed = []
    mae_list_da = []
    mae_list = []
    for i in range(len(region_list)):



        for j in range(len(country_lists[i])):
            data1 = pd.read_csv(
                '../forecasting_using_generated_samples{}/results/results_proposed_{}_LSTM'.format(
                    region_list[i], country_lists[i][j]),
                usecols=['true', 'forecasted']).values[:, 0]

            data2 = pd.read_csv(
                '../forecasting_using_generated_samples{}/results/results_proposed_{}_LSTM'.format(
                    region_list[i], country_lists[i][j]),
                usecols=['true', 'forecasted']).values[:, 1]

            mae_list_proposed.append(np.abs(data1-data2))


            data1 = pd.read_csv(
                '../forecasting_using_generated_samples{}/results/results_basic_{}_LSTM_0.5'.format(
                    region_list[i], country_lists[i][j]),
                usecols=['true', 'forecasted']).values[:, 0]

            data2 = pd.read_csv(
                '../forecasting_using_generated_samples{}/results/results_basic_{}_LSTM_0.5'.format(
                    region_list[i], country_lists[i][j]),
                usecols=['true', 'forecasted']).values[:, 1]

            mae_list_da.append(np.abs(data1 - data2))

            data1 = pd.read_csv(
                '../forecasting_using_generated_samples{}/results/results_basic_{}_LSTM'.format(
                    region_list[i], country_lists[i][j]),
                usecols=['true', 'forecasted']).values[:, 0]

            data2 = pd.read_csv(
                '../forecasting_using_generated_samples{}/results/results_basic_{}_LSTM'.format(
                    region_list[i], country_lists[i][j]),
                usecols=['true', 'forecasted']).values[:, 1]

            mae_list.append(np.abs(data1 - data2))


    proposed_day = []
    da_day = []
    common_day = []
    for j in range(len(mae_list_proposed)):
        for k in range(len(mae_list_proposed[j])//24):
            proposed_day.append(mae_list_proposed[j][24*k:24*(k+1)])
            da_day.append(mae_list_da[j][24 * k:24 * (k + 1)])
            common_day.append(mae_list[j][24 * k:24 * (k + 1)])

    proposed_day = np.array(proposed_day)
    da_day = np.array(da_day)
    common_day = np.array(common_day)

    colors = ['#2F66AC', '#1E90C1','#81CBBF', '#F7CBCC', '#E25659']  # 蓝色 → 白色 → 红色
    positions = [0, 0.4, 0.5, 0.8, 1.0]  # 蓝色在0%，白色在30%，红色在100%

    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_three_colors',
        list(zip(positions, colors)),  # 将位置和颜色配对
        N=n_bins
    )

    # 创建一个包含白色的新颜色映射
    white_cmap = custom_cmap(np.linspace(0, 1, n_bins))
    white_cmap = np.vstack([np.array([1, 1, 1, 1]), white_cmap])  # 在开头添加白色
    custom_cmap_with_white = LinearSegmentedColormap.from_list('custom_with_white', white_cmap, N=n_bins + 1)

    # 设置阈值
    threshold = 0

    print(da_day.shape)

    # 3. 绘制热力图
    #plt.figure(figsize=(5, 4))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    error_list = [proposed_day, da_day, common_day]
    title_list = ['with Feature Separation', 'only with Synthetic Samples', 'without Synthetic Samples']

    for i in range(3):
        time_points = np.tile(np.arange(24), error_list[i].shape[0])  # 时间点
        errors = error_list[i].flatten()  # 误差值

        hist, xedges, yedges = np.histogram2d(
            time_points, errors,
            bins=[24, 1000],  # 24个时间bin，50个误差bin
            density=True  # 归一化为概率密度
        )

        # 扩展为网格坐标
        xx, yy = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        f = hist

        im = ax[i].pcolormesh(xx, yy, f, shading='auto', cmap=custom_cmap_with_white, vmin=0.12, vmax=2.3)
        ax[i].set_ylim(0, 0.08)
        ax[i].set_xlim(0, 22.5)
        ax[i].plot(np.mean(error_list[i], axis=0), color='crimson', lw=3.5, ls='--')
        ax[i].set_xlabel('Time Index [hour]', fontsize=labelsize)
        ax[i].set_ylabel('nMAE', fontsize=labelsize)
        ax[i].tick_params(axis='both', labelsize=ticksize)
        ax[i].set_title(title_list[i], fontsize=titlesize)

        cbar = plt.colorbar(im, label='Probability Density')
        cbar_ticks = np.linspace(threshold, 2, 5)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{x:.3f}' for x in cbar_ticks])
        plt.ylim(0, 0.06)



    plt.tight_layout()
    plt.show()


#plot_heatbar_ablation_error()

def plot_separation_scatter(titlesize=16, ticksize=16, labelsize=16, weather_type='heatwave'):

    if weather_type == 'heatwave':
        sheet_name_list = [
            '嵌入（德州2023热浪）',
            '嵌入（PJM2023热浪）',
            '嵌入（印度2022热浪）',
            '嵌入（广东2023热浪）'
        ]

        title = 'Heat'
    else:
        sheet_name_list = [
            '嵌入（欧洲2018寒潮）',
            '嵌入（湖南2023寒潮）'
        ]

        title = 'Cold'

    fig, ax = plt.subplots(1, 2, figsize=(7, 4))

    plt.subplots_adjust(
        wspace=0.5,  # 列间距（宽度比例）
        hspace=0.4  # 行间距（高度比例）
    )

    proposed_scatter = []
    da_scatter = []
    common_scatter = []
    for i in range(len(sheet_name_list)):
        data = pd.read_excel('../极端温度实验结果.xlsx', sheet_name=sheet_name_list[i])

        data = data.iloc[2:-1, :].reset_index(drop=True).to_numpy()



        for j in range(3):
            for k in range(data.shape[0]-1):
                proposed_scatter.append(data[k, 6 * j + 1])
                da_scatter.append(data[k, 6 * j + 5])
                common_scatter.append(data[k, 6 * j + 3])


    proposed_scatter = np.array(proposed_scatter)
    da_scatter = np.array(da_scatter)
    common_scatter = np.array(common_scatter)

    ax[0].scatter(proposed_scatter, da_scatter,
                  edgecolor='crimson',
                  alpha=0.7,
                  s=15,
                  linewidth=0.5,
                  facecolors='crimson', label='0% ES')

    ax[0].scatter(proposed_scatter, common_scatter,
                  edgecolor='#3783BB',
                  alpha=0.7,
                  s=15,
                  linewidth=0.5,
                  facecolors='#3783BB', label='100% ES')

    ax[0].set_xlim(0, 0.1)
    ax[0].set_ylim(0, 0.1)
    ax[0].plot([0, 0.1], [0, 0.1],
             'k--', alpha=0.5, linewidth=1)
    ax[0].set_title('Per Dataset nMAE \n under {} Waves'.format(title), fontsize=titlesize)
    ax[0].set_xlabel('nMAE of ESDF', fontsize=labelsize)
    ax[0].set_ylabel('nMAE of Comparison Methods', fontsize=labelsize)
    ax[0].set_xticks([0, 0.05, 0.1])
    ax[0].set_yticks([0, 0.05, 0.1])
    ax[0].tick_params(axis='both', labelsize=labelsize)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)


    proposed_scatter = []
    da_scatter = []
    common_scatter = []
    for i in range(len(sheet_name_list)):
        data = pd.read_excel('../极端温度实验结果.xlsx', sheet_name=sheet_name_list[i])

        data = data.iloc[2:-1, :].reset_index(drop=True).to_numpy()

        for j in range(3):
            for k in range(data.shape[0] - 1):
                proposed_scatter.append(data[k, 6 * j + 2])
                da_scatter.append(data[k, 6 * j + 6])
                common_scatter.append(data[k, 6 * j + 4])

    proposed_scatter = np.array(proposed_scatter)
    da_scatter = np.array(da_scatter)
    common_scatter = np.array(common_scatter)

    ax[1].scatter(proposed_scatter, da_scatter,
                  edgecolor='crimson',
                  alpha=0.7,
                  s=15,
                  linewidth=0.5,
                  facecolors='crimson', label='0% ES')

    ax[1].scatter(proposed_scatter, common_scatter,
                  edgecolor='#3783BB',
                  alpha=0.7,
                  s=15,
                  linewidth=0.5,
                  facecolors='#3783BB', label='100% ES')

    ax[1].set_xlim(0, 0.1)
    ax[1].set_ylim(0, 0.1)
    ax[1].plot([0, 0.1], [0, 0.1],
               'k--', alpha=0.5, linewidth=1)
    ax[1].set_title('Per Dataset nRMSE \n under {} Waves'.format(title), fontsize=titlesize)
    ax[1].set_xlabel('nRMSE of ESDF', fontsize=labelsize)
    ax[1].set_ylabel('nRMSE of Comparison Methods', fontsize=labelsize)
    ax[1].set_xticks([0, 0.05, 0.1])
    ax[1].set_yticks([0, 0.05, 0.1])
    ax[1].tick_params(axis='both', labelsize=labelsize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    ax[0].legend(loc='upper center',
               bbox_to_anchor=(0.7, 0.3),  # 调整垂直位置
               ncol=1,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=False)

    plt.tight_layout()
    plt.show()


#plot_separation_scatter(weather_type='heatwave')
#plot_separation_scatter(weather_type='coldwave')


def plot_separation_bar(titlesize=16, ticksize=16, labelsize=16, weather_type='heatwave'):
    if weather_type == 'heatwave':
        region_list = [
            '德州',
            'PJM',
            '印度',
            '广东'
        ]

        title = 'Heat'
    else:
        region_list = [
            '欧洲',
            '湖南'
        ]

        title = 'Cold'

    ratio_list = ['proposed', '0', '0.1', '0.25', '0.5', '0.75', '1']
    colors = ['#F7B799', '#4489C8', 'skyblue', 'skyblue', 'skyblue',
              'skyblue', 'skyblue']
    labels = ['ESDF', '0%', '10%',
              '25%', '50%', '75%',
              '100%']  # 您可以自定义标签
    #region_list = ['欧洲', '广东', '德州', 'PJM', '湖南', '印度']
    region_name = ['GD', 'PJM', 'Texas', 'India', 'Europe', 'HN']
    model_list = ['MLP', 'LSTM', 'CNN']



    fig, ax = plt.subplots(1, 2, figsize=(7, 4))

    plt.subplots_adjust(
        wspace=0.5,  # 列间距（宽度比例）
        hspace=0.4  # 行间距（高度比例）
    )

    proposed_scatter = [[] for i in range(len(ratio_list))]

    da_scatter = []
    common_scatter = []


    for i in range(len(region_list)):
        for j in range(len(ratio_list)):
            data = pd.read_excel('generation_validation/generation_data_{}.xlsx'.format(ratio_list[j]),
                             sheet_name=region_list[i])

            data = data.iloc[2:-1, :].reset_index(drop=True).to_numpy()


            for p in range(data.shape[0] - 1):
                #proposed_scatter[j].append(data[p, 1])
                proposed_scatter[j].append(data[p, 3])
                #proposed_scatter[j].append(data[p, 5])



    proposed_scatter = np.array(proposed_scatter)
    # 计算平均值和95%置信区间
    means = np.mean(proposed_scatter, axis=1)
    stds = np.std(proposed_scatter, axis=1, ddof=1)
    n = proposed_scatter.shape[1]

    # 计算t分布的临界值（95%置信水平）
    t_value = stats.t.ppf(0.95, n - 1)  # 双尾检验
    confidence_intervals = t_value * stds / np.sqrt(n)

    print(proposed_scatter)
    #bars = plt.bar(range(1, len(ratio_list)+1), means, color='skyblue', edgecolor='black', alpha=0.7)


    bars = ax[0].bar(range(1, len(ratio_list)+1), means, color=colors,
              edgecolor='black', alpha=0.7, yerr=confidence_intervals,
              capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})

    for i, bar in enumerate(bars):
        height = bar.get_height()
        # 在柱子底部内部添加标签
        ax[0].text(bar.get_x() + bar.get_width() / 2., height * 0.1,  # 高度为柱子的10%位置
                 labels[i], ha='center', va='bottom', rotation=90,
                 fontsize=10, fontweight='bold', color='black')

    #ax[0].set_xlim(0, 0.1)
    ax[0].set_ylim(0, 0.065)

    #ax[0].plot([0, 0.1], [0, 0.1],
    #           'k--', alpha=0.5, linewidth=1, label='y=x')
    ax[0].set_title('Overall nMAE \n under {} Waves'.format(title), fontsize=titlesize)
    ax[0].set_xlabel('Comparison Methods', fontsize=labelsize)
    ax[0].set_ylabel('nMAE', fontsize=labelsize)
    ax[0].set_xticks(range(1, 1+len(ratio_list)))
    ax[0].set_xticklabels([])
    ax[0].set_yticks([0, 0.02, 0.04, 0.06])
    ax[0].tick_params(axis='both', labelsize=labelsize)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)



    proposed_scatter = [[] for i in range(len(ratio_list))]
    for i in range(len(region_list)):
        for j in range(len(ratio_list)):
            data = pd.read_excel('generation_validation/generation_data_{}.xlsx'.format(ratio_list[j]),
                                 sheet_name=region_list[i])

            data = data.iloc[2:-1, :].reset_index(drop=True).to_numpy()

            for p in range(data.shape[0] - 1):
                #proposed_scatter[j].append(data[p, 2])
                proposed_scatter[j].append(data[p, 4])
                # proposed_scatter[j].append(data[p, 6])

    proposed_scatter = np.array(proposed_scatter)

    # 计算平均值和95%置信区间
    means = np.mean(proposed_scatter, axis=1)
    stds = np.std(proposed_scatter, axis=1, ddof=1)
    n = proposed_scatter.shape[1]

    # 计算t分布的临界值（95%置信水平）
    t_value = stats.t.ppf(0.95, n - 1)  # 双尾检验
    confidence_intervals = t_value * stds / np.sqrt(n)

    bars = ax[1].bar(range(1, len(ratio_list)+1), means, color=colors,
              edgecolor='black', alpha=0.7, yerr=confidence_intervals,
              capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # 在柱子底部内部添加标签
        ax[1].text(bar.get_x() + bar.get_width() / 2., height * 0.1,  # 高度为柱子的10%位置
                 labels[i], ha='center', va='bottom', rotation=90,
                 fontsize=10, fontweight='bold', color='black')
    #ax[1].set_xlim(0, 0.1)
    ax[1].set_ylim(0, 0.065)
    #ax[1].plot([0, 0.1], [0, 0.1],
    #           'k--', alpha=0.5, linewidth=1, label='y=x')
    ax[1].set_title('Overall nRMSE \n under {} Waves'.format(title), fontsize=titlesize)
    ax[1].set_xlabel('Comparison Methods', fontsize=labelsize)
    ax[1].set_ylabel('nRMSE', fontsize=labelsize)
    ax[1].set_xticks([0, 0.05, 0.1])
    ax[1].set_xticks(range(1, 1+len(ratio_list)))
    ax[1].set_xticklabels([])
    ax[1].set_yticks([0, 0.02, 0.04, 0.06])
    ax[1].tick_params(axis='both', labelsize=labelsize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

#plot_separation_bar(weather_type='coldwave')
#plot_separation_bar(weather_type='heatwave')


def plot_separation_curve(titlesize=16, ticksize=16, labelsize=16, weather_type='heatwave'):
    if weather_type == 'heatwave':
        file_name = 'forecasting_using_generated_samples_Guangdong'
        region_name = 'shantou'
        start_day = 6
        period = 14
        title = 'Heat'
        zoomx1 = 168-18
        xlim = 0.6
        tlim = 0.98
        color1 = 'black'
        color2 = '#C74546'
        color3 = 'blue'
        color4 = '#70CDBE'

    else:
        file_name = 'forecasting_using_generated_samples'
        region_name = 'Lithuania'
        start_day = 18
        period = 14
        title = 'Cold'
        zoomx1 = 48
        xlim = 0.75
        tlim = 1
        color1 = 'black'
        color2 = '#C74546'
        color3 = 'blue'
        color4 = '#70CDBE'

    true = pd.read_csv(
        '../{}/results/results_basic_{}_LSTM'.format(file_name, region_name),
        usecols=['true', 'forecasted']).values[:, 0]

    da = pd.read_csv(
        '../{}/results/results_basic_{}_LSTM_1'.format(file_name, region_name),
        usecols=['true', 'forecasted']).values[:, 1]

    proposed = pd.read_csv(
        '../{}/results/results_proposed_{}_LSTM'.format(file_name, region_name),
        usecols=['true', 'forecasted']).values[:, 1]

    original = pd.read_csv(
        '../{}/results/results_basic_{}_LSTM'.format(file_name, region_name),
        usecols=['true', 'forecasted']).values[:, 1]

    plt.rcParams['legend.handlelength'] = 1.0
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.plot(true[start_day*24:start_day*24+period*24], color=color1, label='True', lw=1.5, ls='--')
    ax.plot(proposed[start_day*24:start_day*24+period*24], color=color2, label='ESDF', lw=1)
    ax.plot(da[start_day*24:start_day*24+period*24], color=color3, label='100% ES', lw=1)
    ax.plot(original[start_day * 24:start_day * 24 + period * 24], color=color4, label='0% ES', lw=1)

    # zoom1
    zoomx2 = zoomx1 + 24
    axins_1 = ax.inset_axes((0.1, 0.1, 0.4, 0.3))
    axins_1.plot(true[start_day*24:start_day*24+period*24], color=color1, lw=2, ls='--')
    axins_1.plot(proposed[start_day*24:start_day*24+period*24], color=color2, lw=2)
    axins_1.plot(da[start_day*24:start_day*24+period*24], color=color3, lw=2)
    axins_1.plot(original[start_day * 24:start_day * 24 + period * 24], color=color4, lw=2)

    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontsize=labelsize)
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty0)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty0)
    xy2 = (tx1, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    #axins_1.set_xticks([60, 65, 70])
    axins_1.set_yticks([0.7, 0.8, 0.9])
    axins_1.xaxis.set_tick_params(labelsize=ticksize)
    axins_1.yaxis.set_tick_params(labelsize=ticksize)
    #bars = plt.bar(range(1, len(ratio_list)+1), means, color='skyblue', edgecolor='black', alpha=0.7)

    ax.legend(loc='upper center',
               bbox_to_anchor=(0.75, 0.3),  # 调整垂直位置
               ncol=2,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=True,
               edgecolor='black',
               columnspacing=0.4)

    #ax[0].set_xlim(0, 0.1)
    #ax[0].set_ylim(0, 0.1)

    #ax[0].plot([0, 0.1], [0, 0.1],
    #           'k--', alpha=0.5, linewidth=1, label='y=x')
    ax.set_title('Forecasting Curve under {} Waves'.format(title), fontsize=titlesize)
    ax.set_xlabel('Time Index [hour]', fontsize=labelsize)
    ax.set_ylabel('Norm. Load', fontsize=labelsize)
    #ax.set_xticks(range(1, 1+len(ratio_list)))
    #ax.set_xticklabels([])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    plt.tight_layout()
    plt.show()



#plot_separation_curve(weather_type='heatwave')
#plot_separation_curve(weather_type='coldwave')

def plot_overall_scatter(titlesize=16, ticksize=14, labelsize=16, weather_type='heatwave'):
    if weather_type == 'heatwave':
        file_name = 'forecasting_using_generated_samples_Guangdong'
        region_name = 'zhuhai'
        start_day = 6
        period = 14
        title = 'Heat'
        zoomx1 = 168
        xlim = 0.8
        tlim = 0.98
        color1 = 'black'
        color2 = '#C74546'
        color3 = 'blue'
        color4 = '#70CDBE'
        sheet_name_list = [
            '基础（德州2023热浪）',
            '基础（PJM2023热浪）',
            '基础（印度2022热浪）',
            '基础（广东2023热浪）'
        ]

    else:
        file_name = 'forecasting_using_generated_samples'
        region_name = 'France'
        start_day = 17
        period = 14
        title = 'Cold'
        zoomx1 = 72
        xlim = 0.75
        tlim = 1
        color1 = 'black'
        color2 = '#C74546'
        color3 = 'blue'
        color4 = '#70CDBE'
        sheet_name_list = [
            '基础（欧洲2018寒潮）',
            '基础（湖南2023寒潮）'
        ]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.5)


    title_list = [
        'Europe Coldwave',
        'Texas Heatwave',
        'PJM Heatwave',
        'India Heatwave',
        'Guangdong Heatwave',
        'Hunan Coldwave'
    ]


    colors = ['#0ddbf5', '#1d9bf7', '#8386fc', '#303cf9', '#fe5357', '#fd7c1a', '#ffbd15', '#fcff07']
    models = ['MLP', 'LSTM', 'CNN', 'NBEATS', 'ImpactNet', 'Informer', 'Autoformer']


    proposed = []
    mlp = []
    lstm = []
    cnn = []
    nbeats = []
    impactnet = []
    informer = []
    autoformer = []
    for i in range(len(sheet_name_list)):
        data = pd.read_excel('../极端温度实验结果.xlsx', sheet_name=sheet_name_list[i])

        data = data.iloc[2:-1, :].reset_index(drop=True).to_numpy()
        for j in range(data.shape[0]-1):
            proposed.append(data[j, 1])
            mlp.append(data[j, 3])
            lstm.append(data[j, 5])
            cnn.append(data[j, 7])
            nbeats.append(data[j, 9])
            impactnet.append(data[j, 11])
            informer.append(data[j, 13])
            autoformer.append(data[j, 15])


    comparison = [mlp, lstm, cnn, nbeats, impactnet, informer, autoformer]
    for i in range(len(comparison)):
        ax[0].scatter(proposed, comparison[i],
                  edgecolor=colors[i],
                  alpha=1,
                  s=30,
                  linewidth=1,
                  facecolors=colors[i], label=models[i])

    ax[0].plot([0, 0.1], [0, 0.1],
            'k--', alpha=0.5, linewidth=1, label='y=x')

    # 定义对称中心点
    center = (0.07, 0.07)

    # y=x的斜率是1，垂直线的斜率是-1
    # 箭头长度
    center_x = 0.07
    center_y = 0.07
    arrow_length = 0.03
    length = 0.03
    offset = length / np.sqrt(3)
    text_offset = 0.01  # 文字额外偏移


    ax[0].annotate('Proposed\nbetter',
                xy=(center_x - offset, center_y + offset),  # 箭头终点
                xytext=(center_x - offset + text_offset, center_y + offset - text_offset),  # 文字位置（向右下偏移）
                fontsize=ticksize, ha='center', va='center', color=color1,
                arrowprops=dict(color=color1, arrowstyle='->', lw=2))

    # 第二个箭头：Baselines better（蓝色）
    ax[0].annotate('Baselines\nbetter',
                xy=(center_x + offset, center_y - offset),  # 箭头终点
                xytext=(center_x + offset - text_offset, center_y - offset + text_offset),  # 文字位置（向左上偏移）
                fontsize=ticksize, ha='center', va='center', color=color2,
                arrowprops=dict(color=color2, arrowstyle='->', lw=2))

    ax[0].set_ylim(0, 0.1)
    ax[0].set_xlim(0, 0.1)


    ax[0].tick_params(axis='both', labelsize=ticksize)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[0].set_title('Per Dataset Comparison with Baseline Models \n under {} Waves'.format(title), fontsize=titlesize)
    ax[0].set_xlabel('nMAE of the Proposed Methods', fontsize=labelsize)
    ax[0].set_ylabel('nMAE of Baseline Methods', fontsize=labelsize)





    proposed = []
    mlp = []
    lstm = []
    cnn = []
    nbeats = []
    impactnet = []
    informer = []
    autoformer = []
    for i in range(len(sheet_name_list)):
        data = pd.read_excel('../极端温度实验结果.xlsx', sheet_name=sheet_name_list[i])

        data = data.iloc[2:-1, :].reset_index(drop=True).to_numpy()
        for j in range(data.shape[0] - 1):
            proposed.append(data[j, 2])
            mlp.append(data[j, 4])
            lstm.append(data[j, 6])
            cnn.append(data[j, 8])
            nbeats.append(data[j, 10])
            impactnet.append(data[j, 12])
            informer.append(data[j, 14])
            autoformer.append(data[j, 16])

    comparison = [mlp, lstm, cnn, nbeats, impactnet, informer, autoformer]
    for i in range(len(comparison)):
        ax[1].scatter(proposed, comparison[i],
                      edgecolor=colors[i],
                      alpha=1,
                      s=30,
                      linewidth=1,
                      facecolors=colors[i], label=models[i])

    ax[1].plot([0, 0.1], [0, 0.1],
               'k--', alpha=0.5, linewidth=1, label='y=x')

    # 定义对称中心点
    center = (0.07, 0.07)

    # y=x的斜率是1，垂直线的斜率是-1
    # 箭头长度
    center_x = 0.07
    center_y = 0.07
    arrow_length = 0.03
    length = 0.03
    offset = length / np.sqrt(3)
    text_offset = 0.01  # 文字额外偏移

    ax[1].annotate('Proposed\nbetter',
                   xy=(center_x - offset, center_y + offset),  # 箭头终点
                   xytext=(center_x - offset + text_offset, center_y + offset - text_offset),  # 文字位置（向右下偏移）
                   fontsize=ticksize, ha='center', va='center', color=color1,
                   arrowprops=dict(color=color1, arrowstyle='->', lw=2))

    # 第二个箭头：Baselines better（蓝色）
    ax[1].annotate('Baselines\nbetter',
                   xy=(center_x + offset, center_y - offset),  # 箭头终点
                   xytext=(center_x + offset - text_offset, center_y - offset + text_offset),  # 文字位置（向左上偏移）
                   fontsize=ticksize, ha='center', va='center', color=color2,
                   arrowprops=dict(color=color2, arrowstyle='->', lw=2))

    ax[1].set_ylim(0, 0.1)
    ax[1].set_xlim(0, 0.1)

    ax[1].tick_params(axis='both', labelsize=ticksize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    ax[1].set_title('Per Dataset Comparison with Baseline Models \n under {} Waves'.format(title), fontsize=titlesize)
    ax[1].set_xlabel('nRMSE of the Proposed Methods', fontsize=labelsize)
    ax[1].set_ylabel('nRMSE of Baseline Methods', fontsize=labelsize)


    ax[0].legend(loc='upper center',
              bbox_to_anchor=(0.75, 0.3),  # 调整垂直位置
              ncol=2,  # 设置列数以水平排列
              fontsize=ticksize,
              frameon=True,
              edgecolor='black')

    ax[1].legend(loc='upper center',
                 bbox_to_anchor=(0.75, 0.3),  # 调整垂直位置
                 ncol=2,  # 设置列数以水平排列
                 fontsize=ticksize,
                 frameon=True,
                 edgecolor='black')

    plt.tight_layout()
    plt.show()


#plot_overall_scatter(weather_type='heatwave')
#plot_overall_scatter(weather_type='coldwave')


def plot_scalability_bar(titlesize=12, ticksize=12, labelsize=12, weather_type='heatwave'):

    if weather_type == 'heatwave':


        title = 'Heat'
    else:

        title = 'Cold'

    sheet_name_list = [
        '嵌入（德州2023热浪）',
        '嵌入（PJM2023热浪）',
        '嵌入（印度2022热浪）',
        '嵌入（广东2023热浪）',
        '嵌入（欧洲2018寒潮）',
        '嵌入（湖南2023寒潮）'
    ]

    fig, ax = plt.subplots(2, 3, figsize=(18, 6))
    ax = ax.flatten()

    plt.subplots_adjust(
        wspace=0.5,  # 列间距（宽度比例）
        hspace=0.4  # 行间距（高度比例）
    )


    for i in range(len(sheet_name_list)):

        proposed_scatter = []
        da_scatter = []
        common_scatter = []


        data = pd.read_excel('../极端温度实验结果.xlsx', sheet_name=sheet_name_list[i])
        if i==2:
            print(data)


        data = data.iloc[1:-1, :].reset_index(drop=True).to_numpy()

        if i==2:
            print(data)

        for j in range(3):
            proposed = []
            da = []
            common = []

            for k in range(data.shape[0]):
                proposed.append(data[k, 6 * j + 1])
                da.append(data[k, 6 * j + 5])
                common.append(data[k, 6 * j + 3])

            proposed_scatter.append(proposed)
            da_scatter.append(da)
            common_scatter.append(common)





        common_means = [np.mean(sublist) for sublist in common_scatter]
        da_means = [np.mean(sublist) for sublist in da_scatter]
        proposed_means = [np.mean(sublist) for sublist in proposed_scatter]

        x = np.arange(3)  # 3个组的位置：0, 1, 2
        width = 0.25  # 柱子的宽度

        for j in range(3):
            # 绘制每组的三根柱子
            ax[i].bar(x - width, [common_means[j]], width, label='Common', color='blue')
            ax[i].bar(x, [da_means[j]], width, label='DA', color='orange')
            ax[i].bar(x + width, [proposed_means[j]], width, label='Proposed', color='green')




    plt.tight_layout()
    plt.show()


#plot_scalability_bar()


def plot_test_convergence_heatwave(titlesize=16, ticksize=16, labelsize=16, zoomx1=500, xlim=0.02, tlim=0.07):
    region_list = ['_Guangdong']
    color1 = '#699ECA'
    color2 = '#F16E65'
    color3 = '#008A89'

    country_lists = [['dongguan', 'foshan', 'guangzhou', 'heyuan', 'huizhou',
                      'jiangmen', 'jieyang', 'maoming', 'meizhou', 'qingyuan', 'shantou',
                      'shanwei', 'shaoguan', 'shenzhen', 'yangjiang', 'yunfu', 'zhanjiang',
                      'zhaoqing', 'zhongshan', 'zhuhai']
                     ]

    convergence_lists_proposed = []
    convergence_lists_basic = []
    convergence_lists_rd = []

    for i in range(len(region_list)):
        convergence_list_proposed = []
        convergence_list_basic = []
        convergence_list_rd = []
        for j in range(len(country_lists[i])):
            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/test_basic_{}_LSTM_0.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values
            convergence_list_basic.append(data)

            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/test_basic_{}_LSTM_1.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values
            convergence_list_rd.append(data)

            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/test_proposed_{}_LSTM.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values
            convergence_list_proposed.append(data)

        convergence_lists_basic.append(convergence_list_basic)
        convergence_lists_rd.append(convergence_list_rd)
        convergence_lists_proposed.append(convergence_list_proposed)

    #convergence_lists_ANN = np.array(convergence_lists_ANN)
    #convergence_lists_LSTM = np.array(convergence_lists_LSTM)
    #convergence_lists_CNN = np.array(convergence_lists_CNN)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5))

    # 对于 basic 数据
    flatten_basic = [item for sublist in convergence_lists_basic for item in sublist]
    # 先过滤掉长度小于300的列表
    filtered_basic_lists = [sublist for sublist in flatten_basic if len(sublist) >= 300]
    # 再统一截取到最小长度
    if filtered_basic_lists:
        min_length_basic = min(len(sublist) for sublist in filtered_basic_lists)
        trimmed_basic_lists = [sublist[:min_length_basic] for sublist in filtered_basic_lists]
    else:
        trimmed_basic_lists = []

    # 对于 rd 数据
    flatten_rd = [item for sublist in convergence_lists_rd for item in sublist]
    filtered_rd_lists = [sublist for sublist in flatten_rd if len(sublist) >= 300]
    if filtered_rd_lists:
        min_length_rd = min(len(sublist) for sublist in filtered_rd_lists)
        trimmed_rd_lists = [sublist[:min_length_rd] for sublist in filtered_rd_lists]
    else:
        trimmed_rd_lists = []

    # 对于 proposed 数据
    flatten_proposed = [item for sublist in convergence_lists_proposed for item in sublist]
    filtered_proposed_lists = [sublist for sublist in flatten_proposed if len(sublist) >= 300]
    if filtered_proposed_lists:
        min_length_proposed = min(len(sublist) for sublist in filtered_proposed_lists)
        trimmed_proposed_lists = [sublist[:min_length_proposed] for sublist in filtered_proposed_lists]
    else:
        trimmed_proposed_lists = []



    #x = range(100)
    ax.plot(np.mean(np.array(trimmed_basic_lists), axis=0), color=color1, lw=2, label='0% ES')
    #ax.fill_between(x,
    #                np.max(np.array(flatten_ANN), axis=0),
    #                np.min(np.array(flatten_ANN), axis=0),
    #                color=color1, alpha=0.2)

    ax.plot(np.mean(np.array(trimmed_rd_lists), axis=0), color=color2, lw=2, label='100% ES')
    #ax.fill_between(x,
    #                np.max(np.array(flatten_LSTM), axis=0),
    #                np.min(np.array(flatten_LSTM), axis=0),
    #                color=color2, alpha=0.2)

    ax.plot(np.mean(np.array(trimmed_proposed_lists), axis=0), color=color3, lw=2, label='ESDF')
    #ax.fill_between(x,
    #                np.max(np.array(flatten_CNN), axis=0),
    #                np.min(np.array(flatten_CNN), axis=0),
    #                color=color3, alpha=0.2)

    ax.set_ylabel('nMAE on the Test Set', fontsize=labelsize)
    ax.set_xlabel('Training Epoch', fontsize=labelsize)
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)

    plt.legend(loc='upper center',
               bbox_to_anchor=(0.4, 1),  # 调整垂直位置
               ncol=3,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=False)


    # zoom1
    zoomx2 = zoomx1 + 100
    axins_1 = ax.inset_axes((0.6, 0.4, 0.3, 0.35))
    axins_1.plot(np.mean(np.array(trimmed_basic_lists), axis=0), color=color1, lw=2)
    axins_1.plot(np.mean(np.array(trimmed_rd_lists), axis=0), color=color2, lw=2)
    axins_1.plot(np.mean(np.array(trimmed_proposed_lists), axis=0), color=color3, lw=2)

    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontsize=labelsize)
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty1)
    xy2 = (tx0, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty1)
    xy2 = (tx1, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    axins_1.set_xticks([zoomx1, zoomx1+50, zoomx1+100])
    axins_1.set_yticks([xlim, xlim+0.02, xlim+0.04])
    axins_1.xaxis.set_tick_params(labelsize=ticksize)
    axins_1.yaxis.set_tick_params(labelsize=ticksize)

    # zoom2
    axins_2 = ax.inset_axes((0.1, 0.4, 0.3, 0.35))
    axins_2.plot(np.mean(np.array(trimmed_basic_lists), axis=0), color=color1, lw=2)
    axins_2.plot(np.mean(np.array(trimmed_rd_lists), axis=0), color=color2, lw=2)
    axins_2.plot(np.mean(np.array(trimmed_proposed_lists), axis=0), color=color3, lw=2)

    axins_2.set_xlim(zoomx1-350, zoomx2-350)
    axins_2.set_ylim(xlim, tlim)
    axins_2.set_title('Zoom in', fontsize=labelsize)
    tx0 = zoomx1 - 350
    tx1 = zoomx2 - 350
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty1)
    xy2 = (tx0, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_2, axesB=ax)
    con.set_color('silver')
    axins_2.add_artist(con)
    xy = (tx1, ty1)
    xy2 = (tx1, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_2, axesB=ax)
    con.set_color('silver')
    axins_2.add_artist(con)

    axins_2.set_xticks([tx0, tx0 + 50, tx0 + 100])
    axins_2.set_yticks([xlim, xlim + 0.02, xlim + 0.04])
    axins_2.xaxis.set_tick_params(labelsize=ticksize)
    axins_2.yaxis.set_tick_params(labelsize=ticksize)

    plt.title('Convergence Curve on Heat Wave Datasets', fontsize=titlesize)

    ax.set_ylim(0, 0.2)
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()


#plot_test_convergence_heatwave()

def plot_test_convergence_coldwave(titlesize=16, ticksize=16, labelsize=16, zoomx1=550, xlim=0.02, tlim=0.07):

    region_list = ['']
    color1 = '#699ECA'
    color2 = '#F16E65'
    color3 = '#008A89'

    country_lists = [['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland']
                      ]

    convergence_lists_proposed = []
    convergence_lists_basic = []
    convergence_lists_rd = []

    for i in range(len(region_list)):
        convergence_list_proposed = []
        convergence_list_basic = []
        convergence_list_rd = []
        for j in range(len(country_lists[i])):
            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/test_basic_{}_LSTM_0.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values
            convergence_list_basic.append(data)

            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/test_basic_{}_LSTM_1.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values
            convergence_list_rd.append(data)

            data = pd.read_excel(
                '../forecasting_using_generated_samples{}/Convergence_curve/test_proposed_{}_LSTM.xlsx'.format(
                    region_list[i], country_lists[i][j])).iloc[:, 1].values
            convergence_list_proposed.append(data)

        convergence_lists_basic.append(convergence_list_basic)
        convergence_lists_rd.append(convergence_list_rd)
        convergence_lists_proposed.append(convergence_list_proposed)

    #convergence_lists_ANN = np.array(convergence_lists_ANN)
    #convergence_lists_LSTM = np.array(convergence_lists_LSTM)
    #convergence_lists_CNN = np.array(convergence_lists_CNN)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5))

    # 对于 basic 数据
    flatten_basic = [item for sublist in convergence_lists_basic for item in sublist]
    # 先过滤掉长度小于300的列表
    filtered_basic_lists = [sublist for sublist in flatten_basic if len(sublist) >= 300]
    # 再统一截取到最小长度
    if filtered_basic_lists:
        min_length_basic = min(len(sublist) for sublist in filtered_basic_lists)
        trimmed_basic_lists = [sublist[:min_length_basic] for sublist in filtered_basic_lists]
    else:
        trimmed_basic_lists = []

    # 对于 rd 数据
    flatten_rd = [item for sublist in convergence_lists_rd for item in sublist]
    filtered_rd_lists = [sublist for sublist in flatten_rd if len(sublist) >= 300]
    if filtered_rd_lists:
        min_length_rd = min(len(sublist) for sublist in filtered_rd_lists)
        trimmed_rd_lists = [sublist[:min_length_rd] for sublist in filtered_rd_lists]
    else:
        trimmed_rd_lists = []

    # 对于 proposed 数据
    flatten_proposed = [item for sublist in convergence_lists_proposed for item in sublist]
    filtered_proposed_lists = [sublist for sublist in flatten_proposed if len(sublist) >= 300]
    if filtered_proposed_lists:
        min_length_proposed = min(len(sublist) for sublist in filtered_proposed_lists)
        trimmed_proposed_lists = [sublist[:min_length_proposed] for sublist in filtered_proposed_lists]
    else:
        trimmed_proposed_lists = []


    #x = range(100)
    ax.plot(np.mean(np.array(trimmed_basic_lists), axis=0), color=color1, lw=2, label='0% ES')
    #ax.fill_between(x,
    #                np.max(np.array(flatten_ANN), axis=0),
    #                np.min(np.array(flatten_ANN), axis=0),
    #                color=color1, alpha=0.2)

    ax.plot(np.mean(np.array(trimmed_rd_lists), axis=0), color=color2, lw=2, label='100% ES')
    #ax.fill_between(x,
    #                np.max(np.array(flatten_LSTM), axis=0),
    #                np.min(np.array(flatten_LSTM), axis=0),
    #                color=color2, alpha=0.2)

    ax.plot(np.mean(np.array(trimmed_proposed_lists), axis=0), color=color3, lw=2, label='ESDF')
    #ax.fill_between(x,
    #                np.max(np.array(flatten_CNN), axis=0),
    #                np.min(np.array(flatten_CNN), axis=0),
    #                color=color3, alpha=0.2)

    ax.set_ylabel('nMAE on the Test Set', fontsize=labelsize)
    ax.set_xlabel('Training Epoch', fontsize=labelsize)
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)

    plt.legend(loc='upper center',
               bbox_to_anchor=(0.4, 1),  # 调整垂直位置
               ncol=3,  # 设置列数以水平排列
               fontsize=ticksize,
               frameon=False)


    # zoom1
    zoomx2 = zoomx1 + 100
    axins_1 = ax.inset_axes((0.6, 0.4, 0.3, 0.35))
    axins_1.plot(np.mean(np.array(trimmed_basic_lists), axis=0), color=color1, lw=2)
    axins_1.plot(np.mean(np.array(trimmed_rd_lists), axis=0), color=color2, lw=2)
    axins_1.plot(np.mean(np.array(trimmed_proposed_lists), axis=0), color=color3, lw=2)

    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontsize=labelsize)
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty1)
    xy2 = (tx0, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty1)
    xy2 = (tx1, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    axins_1.set_xticks([zoomx1, zoomx1+50, zoomx1+100])
    axins_1.set_yticks([xlim, xlim+0.02, xlim+0.04])
    axins_1.xaxis.set_tick_params(labelsize=ticksize)
    axins_1.yaxis.set_tick_params(labelsize=ticksize)

    # zoom2
    axins_2 = ax.inset_axes((0.1, 0.4, 0.3, 0.35))
    axins_2.plot(np.mean(np.array(trimmed_basic_lists), axis=0), color=color1, lw=2)
    axins_2.plot(np.mean(np.array(trimmed_rd_lists), axis=0), color=color2, lw=2)
    axins_2.plot(np.mean(np.array(trimmed_proposed_lists), axis=0), color=color3, lw=2)

    axins_2.set_xlim(zoomx1-400, zoomx2-400)
    axins_2.set_ylim(xlim, tlim)
    axins_2.set_title('Zoom in', fontsize=labelsize)
    tx0 = zoomx1 - 400
    tx1 = zoomx2 - 400
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty1)
    xy2 = (tx0, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_2, axesB=ax)
    con.set_color('silver')
    axins_2.add_artist(con)
    xy = (tx1, ty1)
    xy2 = (tx1, ty0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_2, axesB=ax)
    con.set_color('silver')
    axins_2.add_artist(con)

    axins_2.set_xticks([tx0,tx0 + 50, tx0 + 100])
    axins_2.set_yticks([xlim, xlim + 0.02, xlim + 0.04])
    axins_2.xaxis.set_tick_params(labelsize=ticksize)
    axins_2.yaxis.set_tick_params(labelsize=ticksize)

    plt.title('Convergence Curve on Cold Wave Datasets', fontsize=titlesize)

    ax.set_ylim(0, 0.2)
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()

#plot_test_convergence_coldwave()


def ablation_on_classifier(country='Belgium', titlesize=12, ticksize=12, labelsize=12):
    if country == 'Belgium':
        start_day = 14
        period = 15
        color1 = 'black'
        color2 = '#C74546'
        color3 = 'royalblue'
        color4 = '#70CDBE'
        zoomx1 = 144
        xlim = 0.75
        tlim = 1

    else:
        start_day = 14
        period = 15
        color1 = 'black'
        color2 = '#C74546'
        color3 = 'royalblue'
        color4 = '#70CDBE'
        zoomx1 = 144+4*24
        xlim = 0.6
        tlim = 1


    def MAE(x1, x2):
        lst = np.array([abs(x1[i] - x2[i]) for i in range(x1.shape[0])])
        return np.mean(lst)


    true = pd.read_csv(
        'Impact of deviation/results_proposed_{}_conditional'.format(country),
        usecols=['true', 'forecasted']).values[:, 0]

    conditional = pd.read_csv(
        'Impact of deviation/results_proposed_{}_conditional'.format(country),
        usecols=['true', 'forecasted']).values[:, 1]

    unconditional = pd.read_csv(
        'Impact of deviation/results_proposed_{}_unconditional'.format(country),
        usecols=['true', 'forecasted']).values[:, 1]

    print(MAE(true[24 * start_day:24 * (start_day + period)], conditional[24 * start_day:24 * (start_day + period)]))
    print(MAE(true[24 * start_day:24 * (start_day + period)], unconditional[24 * start_day:24 * (start_day + period)]))

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(true[start_day * 24:start_day * 24 + period * 24], color=color1, label='True', lw=2, ls='--')
    ax.plot(conditional[start_day * 24:start_day * 24 + period * 24], color=color2, label='Conditional', lw=1.5)
    ax.plot(unconditional[start_day * 24:start_day * 24 + period * 24], color=color3, label='Unconditional', lw=1.5)


    # zoom1
    zoomx2 = zoomx1 + 24
    axins_1 = ax.inset_axes((0.1, 0.1, 0.4, 0.3))
    axins_1.plot(true[start_day * 24:start_day * 24 + period * 24], color=color1, lw=2, ls='--')
    axins_1.plot(conditional[start_day * 24:start_day * 24 + period * 24], color=color2, lw=2)
    axins_1.plot(unconditional[start_day * 24:start_day * 24 + period * 24], color=color3, lw=2)

    axins_1.set_xlim(zoomx1, zoomx2)
    axins_1.set_ylim(xlim, tlim)
    axins_1.set_title('Zoom in', fontsize=labelsize)
    tx0 = zoomx1
    tx1 = zoomx2
    ty0 = xlim
    ty1 = tlim
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    xy = (tx0, ty0)
    xy2 = (tx0, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)
    xy = (tx1, ty0)
    xy2 = (tx1, ty1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins_1, axesB=ax)
    con.set_color('silver')
    axins_1.add_artist(con)

    # axins_1.set_xticks([60, 65, 70])
    axins_1.set_yticks([0.7, 0.8, 0.9])
    axins_1.xaxis.set_tick_params(labelsize=ticksize)
    axins_1.yaxis.set_tick_params(labelsize=ticksize)
    # bars = plt.bar(range(1, len(ratio_list)+1), means, color='skyblue', edgecolor='black', alpha=0.7)

    ax.set_ylim(0, 1.05)
    ax.axvspan(120, 120 + 8 * 24, alpha=0.1, color='lightcoral')
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.75, 0.3),  # 调整垂直位置
              ncol=1,  # 设置列数以水平排列
              fontsize=ticksize,
              frameon=True,
              edgecolor='black')
    ax.set_title(country, fontsize=titlesize)
    ax.set_xlabel('Time Index', fontsize=labelsize)
    ax.set_ylabel('Norm. Load', fontsize=labelsize)
    # ax.set_xticks(range(1, 1+len(ratio_list)))
    # ax.set_xticklabels([])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig('figures/{}_classifier_ablation.pdf'.format(country))
    plt.show()


#ablation_on_classifier('France')

def ablation_on_constraints(country='Europe', titlesize=12, ticksize=12, labelsize=12):
    if country=='Europe':
        color=['#B3E5FC', '#81D4FA', '#4FC3F7', '#29B6F6' ]
        line_color='#0D47A1'
        weather='Cold Waves'
        min_ylim = 0.015
        max_ylim = 0.04
        best_value = 0.03292
        best_value_rmse = 0.04240

    else:
        color=['#F8BBD0', '#F48FB1', '#EC407A', '#D81B60']
        line_color='#880E4F'
        weather = 'Heat Waves'
        min_ylim = 0.015
        max_ylim = 0.05
        best_value = 0.037093
        best_value_rmse = 0.0507373


    benchmark = pd.read_csv(
        'impact of constraints/{}_benchmark.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    no_ort = pd.read_csv(
        'impact of constraints/{}_no_ort.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    no_sim = pd.read_csv(
        'impact of constraints/{}_no_sim.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    proposed = pd.read_csv(
        'impact of constraints/{}_proposed.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    # 计算每组的统计量
    data_groups = [benchmark, no_ort, no_sim, proposed]
    group_names = ['no constraints', 'no orthogonal', 'no similarity', 'proposed']

    means = [np.mean(group) for group in data_groups]

    # 设置图形
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    x_pos = np.arange(len(group_names))
    bar_width = 0.3

    # 绘制柱状图（均值）
    bars = ax[0].bar(x_pos, means, bar_width, label='Mean',
                  color=color, alpha=0.7, edgecolor='black')

    best_index = 3
    # 添加星星标记在最佳柱子上方
    ax[0].plot(x_pos[best_index], means[-1] * 1.03, marker='*', markersize=15,
               color='gold', markeredgecolor='darkorange', markeredgewidth=1)

    # 添加水平虚线表示最佳基准线
    ax[0].axhline(y=best_value, color='red', linestyle='--', alpha=0.7,
                  linewidth=1.5, label='best baseline')


    # 美化图形
    ax[0].set_ylim(min_ylim, max_ylim)
    ax[0].set_xlabel('Settings', fontsize=12)
    ax[0].set_ylabel('nMAE', fontsize=12)
    ax[0].set_title('Ablation on Constraints in RD Network under {}'.format(weather), fontsize=14)
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(group_names)
    ax[0].tick_params(axis='both', labelsize=labelsize)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].legend(fontsize=ticksize)


    benchmark = pd.read_csv(
        'impact of constraints/{}_benchmark.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 1].values

    no_ort = pd.read_csv(
        'impact of constraints/{}_no_ort.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 1].values

    no_sim = pd.read_csv(
        'impact of constraints/{}_no_sim.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 1].values

    proposed = pd.read_csv(
        'impact of constraints/{}_proposed.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 1].values

    # 计算每组的统计量
    data_groups = [benchmark, no_ort, no_sim, proposed]
    group_names = ['no constraints', 'no orthogonal', 'no similarity', 'proposed']

    means = [np.mean(group) for group in data_groups]


    # 绘制柱状图（均值）
    bars = ax[1].bar(x_pos, means, bar_width, label='Mean',
                     color=color, alpha=0.7, edgecolor='black')

    best_index = 3
    # 添加星星标记在最佳柱子上方
    ax[1].plot(x_pos[best_index], means[-1] * 1.03, marker='*', markersize=15,
               color='gold', markeredgecolor='darkorange', markeredgewidth=1)

    # 添加水平虚线表示最佳基准线
    ax[1].axhline(y=best_value_rmse, color='red', linestyle='--', alpha=0.7,
                  linewidth=1.5, label='best baseline')

    # 美化图形
    ax[1].set_ylim(min_ylim+0.01, max_ylim+0.015)
    ax[1].set_xlabel('Settings', fontsize=12)
    ax[1].set_ylabel('nRMSE', fontsize=12)
    ax[1].set_title('Ablation on Constraints in RD Network under {}'.format(weather), fontsize=14)
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels(group_names)
    ax[1].tick_params(axis='both', labelsize=labelsize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].legend(fontsize=ticksize)



    plt.tight_layout()
    plt.savefig('figures/{}_constraints_ablation.pdf'.format(weather))
    plt.show()

#ablation_on_constraints('Europe')
#ablation_on_constraints('PJM')

def ablation_on_proportion(country='Europe', titlesize=12, ticksize=12, labelsize=12):
    if country=='Europe':
        color=['#fee3ce', '#eabaa1', '#dc917b', '#c44438', '#b7282e' ]
        line_color='#0D47A1'
        weather='Cold Waves'
        min_ylim = 0.01
        max_ylim = 0.04
        best_value = 0.03292
        best_value_rmse = 0.04240

    else:
        color=['#fee3ce', '#eabaa1', '#dc917b', '#c44438', '#b7282e' ]
        line_color='#880E4F'
        weather = 'Heat Waves'
        min_ylim = 0.015
        max_ylim = 0.055
        best_value = 0.037093
        best_value_rmse = 0.0507373

    # 设置图形
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))



    p_1_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.1.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    p_2_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.25.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    p_3_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.5.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    p_4_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.75.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    p_5_mae = pd.read_csv(
        'impact of proportion_proposed/{}_1.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-2, 0].values

    # 计算每组的统计量
    data_groups = [p_1_mae, p_2_mae, p_3_mae, p_4_mae, p_5_mae]
    group_names = ['0.10', '0.25', '0.50', '0.75', '1.00']

    x_pos = np.arange(len(group_names))
    bar_width = 0.3

    means = [np.mean(group) for group in data_groups]



    # 绘制柱状图（均值）
    bars = ax[0].bar(x_pos, means, bar_width, label='Mean',
                  color=color, alpha=0.7, edgecolor='black')
        # 假设第五个柱状图（索引为4）是最佳结果
    best_index = 4
    # 添加星星标记在最佳柱子上方
    ax[0].plot(x_pos[best_index], means[-1] * 1.03, marker='*', markersize=15,
               color='gold', markeredgecolor='darkorange', markeredgewidth=1)

    # 添加水平虚线表示最佳基准线
    ax[0].axhline(y=best_value, color='red', linestyle='--', alpha=0.7,
                  linewidth=1.5, label='best baseline')

    # 美化图形
    ax[0].set_ylim(min_ylim, max_ylim)
    ax[0].set_xlabel('Proportion of Synthetic Samples', fontsize=12)
    ax[0].set_ylabel('nMAE', fontsize=12)
    ax[0].set_title('Impacts of Synthetic Samples on ESDF under {}'.format(weather), fontsize=14)
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(group_names)
    ax[0].tick_params(axis='both', labelsize=labelsize)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].legend(fontsize=ticksize)


    # RMSE
    p_1_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.1.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-1, 1].values

    p_2_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.25.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-1, 1].values

    p_3_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.5.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-1, 1].values

    p_4_mae = pd.read_csv(
        'impact of proportion_proposed/{}_0.75.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-1, 1].values

    p_5_mae = pd.read_csv(
        'impact of proportion_proposed/{}_1.csv'.format(country),
        skiprows=1  # 跳过第一行（表头）
    ).iloc[:-1, 1].values

    # 计算每组的统计量
    data_groups = [p_1_mae, p_2_mae, p_3_mae, p_4_mae, p_5_mae]
    group_names = ['0.10', '0.25', '0.50', '0.75', '1.00']

    x_pos = np.arange(len(group_names))
    bar_width = 0.3

    means = [np.mean(group) for group in data_groups]

    # 绘制柱状图（均值）
    bars = ax[1].bar(x_pos, means, bar_width, label='Mean',
                  color=color, alpha=0.7, edgecolor='black')
    best_index = 4
    # 添加星星标记在最佳柱子上方
    ax[1].plot(x_pos[best_index], means[-1] * 1.03, marker='*', markersize=15,
               color='gold', markeredgecolor='darkorange', markeredgewidth=1)
    # 添加水平虚线表示最佳基准线
    ax[1].axhline(y=best_value_rmse, color='red', linestyle='--', alpha=0.7,
                  linewidth=1.5, label='best baseline')
    # 美化图形
    ax[1].set_ylim(min_ylim+0.015, max_ylim+0.015)
    ax[1].set_xlabel('Proportion of Synthetic Samples', fontsize=12)
    ax[1].set_ylabel('nRMSE', fontsize=12)
    ax[1].set_title('Impacts of Synthetic Samples on ESDF under {}'.format(weather), fontsize=14)
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels(group_names)
    ax[1].tick_params(axis='both', labelsize=labelsize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].legend(fontsize=ticksize)

    plt.tight_layout()

    plt.savefig('figures/{}_proportion_ablation.pdf'.format(weather))
    plt.show()

#ablation_on_proportion('Europe')
#ablation_on_proportion('PJM')





