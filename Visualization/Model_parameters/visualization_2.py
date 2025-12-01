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

def basic_reduction():
    fig, ax = plt.subplots(1, 6, figsize=(15, 3.5))

    colors = ['#46AEA0FF'] + ['#98D048FF'] + ['#F8D068FF'] + ['#88A0DCFF'] + \
             ['#F6B8BDFF'] + ['#F8A070FF']
    country_list = ['Guangdong', 'PJM', 'Texas', 'India', 'Hunan', 'Europe']
    mae_proposed_list = [0.02566, 0.029185684, 0.034255125, 0.02625, 0.02853, 0.024812333]
    rmse_proposed_list = [0.03378, 0.039893, 0.043263125, 0.03537, 0.03705, 0.032820389]
    mae_baseline_list = [0.03344, 0.032913947, 0.04211, 0.02914, 0.03227, 0.0331245]
    rmse_baseline_list = [0.04543, 0.045667, 0.05253875, 0.03802, 0.04054, 0.042476944]


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
                          linewidth=1,
                          label='Proposed')

        bars2 = ax[i].bar(x - x_offset / 2, data2,
                          width=bar_width,
                          bottom=data1,
                          color=colors[i], alpha=0.5,
                          edgecolor='black',
                          linestyle='--',
                          linewidth=1,  # 可选：添加填充图案
                          label='Reduction')

        # 添加数值标签
        # ax[i].bar_label(bars1, fmt='%.2f', padding=2)
        #ax[i].bar_label((bars2-bars1)/bars2, fmt='%.2f', padding=2)
        # 计算并添加文本标签
        for j in range(len(data1)):
            if data2[j] != 0:  # 确保不除以零
                value = data2[j]/data1[j]
                ax[i].text(x[j] - x_offset / 2, data1[j] + data2[j] + 0.002, f'{value*100:.1f}'+'%↓',
                           ha='center', va='bottom', color='#C24841FF', fontsize=12)

        # 美化坐标轴
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_ylim(0, 0.085)
        ax[i].set_yticks([0, 0.02, 0.04, 0.06])
        ax[i].set_title(country_list[i])
        ax[i].legend(frameon=False)

        # 设置 x 轴范围以增加柱子与 y 轴之间的距离
        ax[i].set_xlim(-0.75, len(data1) - 0.5)

        # 设置 x 轴刻度和标签
        ax[i].set_xticks(x - x_offset / 2)  # 设置 x 轴刻度位置
        ax[i].set_xticklabels(['nMAE', 'nRMSE'], rotation=45, fontsize=12, ha='center', va='top')  # 设置 x 轴标签
        ax[i].tick_params(axis='both', labelsize=12)

    ax[0].set_ylabel('Forecasting Error', fontsize=12)
    plt.tight_layout()
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


def plot_generated_sample_distribution_coldwave():
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
    # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
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

        coldwave_samples = generate_coldwave_samples(country, num_samples=600).cpu().detach().numpy()
        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)


        load_slice_list = np.array(load_slice_list)
        #print(load_slice_list.shape)
        tem_slice_list = np.array(tem_slice_list)

        sample_load_list = np.mean(load_slice_list[:, :-24], axis=1)
        label_load_list = np.mean(load_slice_list[:, -24:], axis=1)
        sample_tem_list = np.mean(tem_slice_list[:, :-24], axis=1)
        generated_sample_load_list = np.mean(coldwave_samples[:, 0, :-24], axis=1)
        generated_sample_tem_list = np.mean(coldwave_samples[:, 1, :-24], axis=1)

        # 设置统一的bins（确保两个直方图使用相同分箱）
        bins = np.linspace(
            min(min(sample_tem_list), min(generated_sample_tem_list)),
            max(max(sample_tem_list), max(generated_sample_tem_list)),
            30  # 分箱数量
        )

        # 绘制重叠柱状图
        plt.hist(sample_tem_list, bins=bins, alpha=0.7, label='Real Samples', color='blue')
        plt.hist(generated_sample_tem_list, bins=bins, alpha=0.7, label='Generated Samples', color='orange')

        # 添加标注
        plt.xlabel('Temperature Value')
        plt.ylabel('Frequency')
        plt.title('Distribution Comparison: Real vs Generated Samples')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()


plot_generated_sample_distribution_coldwave()