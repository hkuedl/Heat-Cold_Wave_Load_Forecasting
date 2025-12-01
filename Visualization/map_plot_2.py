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

def plot_pjm():
    import geopandas as gpd
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)
    usa_states = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip")
    # 筛选出德州数据

    counties = gpd.read_file("map_data/PJM Zone.zip")
    #tx_counties = counties[counties['STATEFP'] == '10'].copy()
    counties.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)

    plt.show()
    print(counties)




#plot_pjm()

def check_pjm(titlesize=16, ticksize=14, labelsize=14, region='pjm'):
    # 读取数据
    city_name = ['Allegheny Power System',
                 'American Electric Power Co., Inc', 'Atlantic Electric Company',
                 'Baltimore Gas and Electric Company',
                 'ComEd', 'Dayton Power and Light Company', 'Delmarva Power and Light',
                 'Dominion Energy', 'Duke Energy Ohio', 'Duquesne Light',
                 'East Kentucky Power Coop', 'First Energy - Pennsylvania Electric Company',
                 'Jersey Central Power and Light Company', 'Metropolitan Edison Company',
                 'Pennsylvania Electric Company',
                 'Pennsylvania Power and Light Company',
                 'Potomac Electric Power',
                 'Public Service Electric and Gas Company']

    pjm_zones = ['Allegheny Power', 'American Electric Power',
                 'Atlantic Electric', 'Baltimore Gas & Electric Company',
                 'Commonwealth Edison Co.', 'Dayton Power & Light Co.',
                 'Delmarva Power & Light Company', ' Virginia Power Company',
                 'Duke Energy Ohio', 'Duquesne Light Company',
                 'East Kentucky Power Coop.', 'FirstEnergy ATSI',
                 'Jersey Central Power & Light Company', 'Metropolitan Edison Company',
                 'Pennsylvania Electric Company', 'Pennsylvania Power & Light Company',
                 'Potomac Electric Power Company', 'Public Service Electric & Gas Company'
                 ]
    # 读取PJM数据
    pjm = gpd.read_file("map_data/PJM Zone.zip")
    print("原始PJM坐标系:", pjm.crs)

    # 转换为WGS84地理坐标系
    pjm_wgs84 = pjm.to_crs('EPSG:4326')
    print("转换后坐标系:", pjm_wgs84.crs)

    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip").copy()

    # 检查PJM数据的坐标系
    print("PJM坐标系:", pjm_wgs84.crs)

    # 将世界地图转换为PJM数据的坐标系
    world_projected = world.to_crs(pjm_wgs84.crs)

    # 在投影坐标系中绘制
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world_projected.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)

    colors = []

    cmap = plt.get_cmap('Greens')  # 黄-橙-红色标，适用于热力值
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
    hotwave_tem_norm = []
    common_tem_norm = []
    for i in range(len(city_name)):
        data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(city_name[i]), header=0,
                           usecols=['Date_Hour', 'Load', 'Temperature'])

        strat_time = '2021/03/28/00'
        end_time = '2022/09/29/23'
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
        #start_date = pd.to_datetime(strat_time)  ## Thursday
        #end_date = pd.to_datetime(end_time)
        #data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]

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

            if EHF ==0 and EHF == 0:
                common_load_mean.append(load[24 * j: 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

        hot_common_ratio = np.mean(hotwave_load_mean) / np.mean(common_load_mean) - 1
        color = cmap(norm(hot_common_ratio))
        colors.append(color)

        print(hot_common_ratio)
        print(i)

    # 绘制PJM区域（现在坐标单位一致）
    for i in range(len(pjm_zones)):
        zone_name = pjm_zones[i]
        color = colors[i]

        current_zone = pjm_wgs84[pjm_wgs84['Zone_Name'] == zone_name]

        if not current_zone.empty:
            current_zone.plot(ax=ax, color=color, alpha=1,
                              edgecolor='white', linewidth=1.5,
                              label=zone_name)
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
    ax_inset_2.xaxis.set_tick_params(labelsize=ticksize)

    ax.set_xlim(-91, -73)  # 经度范围（西→东）
    ax.set_ylim(32, 42)  # 纬度范围（南→北）
    ax.set_yticks([30, 35, 40])
    ax.set_xticks([-90, -85, -80, -75])
    ax.set_xticklabels(['-90°E', '-85°E', '-80°E', '-75°E'])
    ax.set_yticklabels(['30°N', '35°N', '40°N'])
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
    # plt.axis('equal')
    plt.show()

check_pjm()