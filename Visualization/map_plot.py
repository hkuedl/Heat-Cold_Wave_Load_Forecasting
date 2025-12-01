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


def Texas_map():
    counties = gpd.read_file("map_data/tl_2021_us_county.zip")
    print(counties)
    # 筛选出德州的县，'STATEFP'是州的FIPS代码，德州是'48'
    tx_counties = counties[counties['STATEFP'] == '48'].copy()

    # 2. 构建您提供的ERCOT区域映射字典
    # 这里以您提到的NORTH_WEST为例，请根据您找到的完整信息补充所有区域和县。
    # 字典结构：{'县名': 'ERCOT区域'}
    erlot_region_mapping = {
        'Victoria':'COAST',
        'Calhoun': 'COAST',
        'Jackson': 'COAST',
        'Warton': 'COAST',
        'Matagorda': 'COAST',
        'Fort Bend': 'COAST',
        'Brazoria': 'COAST',
        'Waller': 'COAST',
        'Harris': 'COAST',
        'Galveston': 'COAST',
        'Montgomery': 'COAST',
        'San Jacinto': 'COAST',
        'Liberty': 'COAST',
        'Chambers': 'COAST',
        'Polk': 'COAST',
        'Hardin': 'COAST',
        'Jefferson': 'COAST',
        'Tyler': 'COAST',
        'Jasper': 'COAST',
        'Orange': 'COAST',
        'Newton': 'COAST',
        'Wharton': 'COAST',
        'Hopkins': 'EAST',
        'Franklin': 'EAST',
        'Titus': 'EAST',
        'Morris': 'EAST',
        'Cass': 'EAST',
        'Rains': 'EAST',
        'Wood': 'EAST',
        'Marion': 'EAST',
        'Harrison': 'EAST',
        'Van Zandt': 'EAST',
        'Smith': 'EAST',
        'Gregg': 'EAST',
        'Henderson': 'EAST',
        'Rusk': 'EAST',
        'Panola': 'EAST',
        'Freestone': 'EAST',
        'Anderson': 'EAST',
        'Cherokee': 'EAST',
        'Nacogdoches': 'EAST',
        'Shelby': 'EAST',
        'San Augustine': 'EAST',
        'Sabine': 'EAST',
        'Robertson': 'EAST',
        'Leon': 'EAST',
        'Houston': 'EAST',
        'Angelina': 'EAST',
        'Trinity': 'EAST',
        'Brazos': 'EAST',
        'Madison': 'EAST',
        'Walker': 'EAST',
        'Grimes': 'EAST',
        'Camp': 'EAST',
        'Upshur': 'EAST',
        'El Paso': 'FAR_WEST',
        'Hudspeth': 'FAR_WEST',
        'Culberson': 'FAR_WEST',
        'Reeves': 'FAR_WEST',
        'Loving': 'FAR_WEST',
        'Winkler': 'FAR_WEST',
        'Ward': 'FAR_WEST',
        'Andrews': 'FAR_WEST',
        'Ector': 'FAR_WEST',
        'Crane': 'FAR_WEST',
        'Dawson': 'FAR_WEST',
        'Martin': 'FAR_WEST',
        'Midland': 'FAR_WEST',
        'Upton': 'FAR_WEST',
        'Borden': 'FAR_WEST',
        'Howard': 'FAR_WEST',
        'Glasscock': 'FAR_WEST',
        'Reagan': 'FAR_WEST',
        'Jeff Davis': 'FAR_WEST',
        'Pecos': 'FAR_WEST',
        'Crockett': 'FAR_WEST',
        'Presidio': 'FAR_WEST',
        'Brewster': 'FAR_WEST',
        'Terrell': 'FAR_WEST',
        'Cooke': 'NORTH',
        'Grayson': 'NORTH',
        'Fannin': 'NORTH',
        'Lamar': 'NORTH',
        'Red River': 'NORTH',
        'Bowie': 'NORTH',
        'Wheeler': 'NORTH',
        'Donley': 'NORTH',
        'Collingsworth': 'NORTH',
        'Brisco': 'NORTH',
        'Hall': 'NORTH',
        'Childress': 'NORTH',
        'Hardeman': 'NORTH',
        'Floyd': 'NORTH',
        'Motley': 'NORTH',
        'Cottle': 'NORTH',
        'Foard': 'NORTH',
        'Wilbarger': 'NORTH',
        'Wichita': 'NORTH',
        'Clay': 'NORTH',
        'Montague': 'NORTH',
        'Crosby': 'NORTH',
        'Dickens': 'NORTH',
        'King': 'NORTH',
        'Knox': 'NORTH',
        'Baylor': 'NORTH',
        'Archer': 'NORTH',
        'Garza': 'NORTH',
        'Kent': 'NORTH',
        'Stonewall': 'NORTH',
        'Haskell': 'NORTH',
        'Throckmorton': 'NORTH_CENTRAL',
        'Young': 'NORTH_CENTRAL',
        'Jack': 'NORTH_CENTRAL',
        'Wise': 'NORTH_CENTRAL',
        'Denton': 'NORTH_CENTRAL',
        'Collin': 'NORTH_CENTRAL',
        'Hunt': 'NORTH_CENTRAL',
        'Delta': 'NORTH_CENTRAL',
        'Shackelford': 'NORTH_CENTRAL',
        'Stephens': 'NORTH_CENTRAL',
        'Palo Pinto': 'NORTH_CENTRAL',
        'Parker': 'NORTH_CENTRAL',
        'Tarrant': 'NORTH_CENTRAL',
        'Dallas': 'NORTH_CENTRAL',
        'Rockwall': 'NORTH_CENTRAL',
        'Kaufman': 'NORTH_CENTRAL',
        'Callahan': 'NORTH_CENTRAL',
        'Eastland': 'NORTH_CENTRAL',
        'Erath': 'NORTH_CENTRAL',
        'Hood': 'NORTH_CENTRAL',
        'Somervell': 'NORTH_CENTRAL',
        'Johnson': 'NORTH_CENTRAL',
        'Ellis': 'NORTH_CENTRAL',
        'Brown': 'NORTH_CENTRAL',
        'Comanche': 'NORTH_CENTRAL',
        'Hamilton': 'NORTH_CENTRAL',
        'Bosque': 'NORTH_CENTRAL',
        'Hill': 'NORTH_CENTRAL',
        'Navarro': 'NORTH_CENTRAL',
        'Coryell': 'NORTH_CENTRAL',
        'McLennan': 'NORTH_CENTRAL',
        'Limestone': 'NORTH_CENTRAL',
        'Bell': 'NORTH_CENTRAL',
        'Falls': 'NORTH_CENTRAL',
        'Mills': 'NORTH_CENTRAL',
        'Austin': 'SOUTH_CENTRAL',
        'Bandera': 'SOUTH_CENTRAL',
        'Bastrop': 'SOUTH_CENTRAL',
        'Bexar': 'SOUTH_CENTRAL',
        'Blanco': 'SOUTH_CENTRAL',
        'Burleson': 'SOUTH_CENTRAL',
        'Burnet': 'SOUTH_CENTRAL',
        'Caldwell': 'SOUTH_CENTRAL',
        'Colorado': 'SOUTH_CENTRAL',
        'Comal': 'SOUTH_CENTRAL',
        'DeWitt': 'SOUTH_CENTRAL',
        'Fayette': 'SOUTH_CENTRAL',
        'Gonzales': 'SOUTH_CENTRAL',
        'Guadalupe': 'SOUTH_CENTRAL',
        'Hays': 'SOUTH_CENTRAL',
        'Karnes': 'SOUTH_CENTRAL',
        'Kendall': 'SOUTH_CENTRAL',
        'Lavaca': 'SOUTH_CENTRAL',
        'Lee': 'SOUTH_CENTRAL',
        'Medina': 'SOUTH_CENTRAL',
        'Milam': 'SOUTH_CENTRAL',
        'Travis': 'SOUTH_CENTRAL',
        'Washington': 'SOUTH_CENTRAL',
        'Williamson': 'SOUTH_CENTRAL',
        'Wilson': 'SOUTH_CENTRAL',
        'Aransas': 'SOUTHERN',
        'Atascosa': 'SOUTHERN',
        'Bee': 'SOUTHERN',
        'Brooks': 'SOUTHERN',
        'Cameron': 'SOUTHERN',
        'Dimmit': 'SOUTHERN',
        'Duval': 'SOUTHERN',
        'Frio': 'SOUTHERN',
        'Goliad': 'SOUTHERN',
        'Hidalgo': 'SOUTHERN',
        'Jim Hogg': 'SOUTHERN',
        'Jim Wells': 'SOUTHERN',
        'Kenedy': 'SOUTHERN',
        'Kleberg': 'SOUTHERN',
        'La Salle': 'SOUTHERN',
        'Live Oak': 'SOUTHERN',
        'Maverick': 'SOUTHERN',
        'McMullen': 'SOUTHERN',
        'Nueces': 'SOUTHERN',
        'Refugio': 'SOUTHERN',
        'San Patricio': 'SOUTHERN',
        'Starr': 'SOUTHERN',
        'Webb': 'SOUTHERN',
        'Willacy': 'SOUTHERN',
        'Zapata': 'SOUTHERN',
        'Zavala': 'SOUTHERN',
        'Coke': 'WEST',
        'Coleman': 'WEST',
        'Concho': 'WEST',
        'Edwards': 'WEST',
        'Fisher': 'WEST',
        'Gillespie': 'WEST',
        'Irion': 'WEST',
        'Jones': 'WEST',
        'Kerr': 'WEST',
        'Kimble': 'WEST',
        'Kinney': 'WEST',
        'Lampasas': 'WEST',
        'Llano': 'WEST',
        'Mason': 'WEST',
        'McCulloch': 'WEST',
        'Menard': 'WEST',
        'Mitchell': 'WEST',
        'Nolan': 'WEST',
        'Real': 'WEST',
        'Runnels': 'WEST',
        'San Saba': 'WEST',
        'Schleicher': 'WEST',
        'Scurry': 'WEST',
        'Sterling': 'WEST',
        'Sutton': 'WEST',
        'Taylor': 'WEST',
        'Tom Green': 'WEST',
        'Uvalde': 'WEST',
        'Val Verde': 'WEST'
    }

    # 将映射字典转换为一个DataFrame，方便后续合并
    mapping_df = pd.DataFrame(list(erlot_region_mapping.items()), columns=['NAME', 'ERCOT_Region'])

    # 3. 将ERCOT区域信息合并到德州地理数据中
    # 使用左连接，确保所有德州县都被保留，即使有些县不在您的映射里（这些县区域会为NaN）
    tx_counties_with_region = tx_counties.merge(mapping_df, on='NAME', how='left')
    regions_gdf = tx_counties_with_region.dissolve(by='ERCOT_Region')

    print(regions_gdf)

    # 检查是否有县未匹配上映射关系
    print("未在映射关系中找到的县：")
    print(tx_counties_with_region[tx_counties_with_region['ERCOT_Region'].isna()]['NAME'].tolist())

    # 4. 绘制地图
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # 按ERCOT区域着色绘制各县
    # 为每个区域指定特定颜色
    region_colors = {
        'NORTH_CENTRAL': '#1f77b4',
        'NORTH_WEST': '#ff7f0e',
        'COAST': '#2ca02c',
        # 为其他区域添加颜色...
    }

    # 然后使用这些颜色
    colors = [region_colors.get(region, 'gray') for region in regions_gdf.index]
    regions_gdf.plot(ax=ax,
                     color=colors,
                     edgecolor='white',
                     linewidth=1.5,
                     alpha=0.7)

    # 添加标题，美化图表
    ax.set_title('ERCOT Regions of Texas (by County)', fontsize=16)
    ax.set_axis_off()  # 不显示坐标轴
    plt.tight_layout()
    plt.show()


#Texas_map()


def plot_texas_map(titlesize=16, ticksize=14, labelsize=14, region='texas'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path



    city_name = ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C',
                     'SOUTH_C', 'SOUTHERN', 'WEST', 'FAR_WEST']

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)
    usa_states = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip")
    # 筛选出德州数据



    cmap = plt.get_cmap('YlOrRd')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)

    counties = gpd.read_file("map_data/tl_2021_us_county.zip")
    # 筛选出德州的县，'STATEFP'是州的FIPS代码，德州是'48'
    tx_counties = counties[counties['STATEFP'] == '48'].copy()
    erlot_region_mapping = {
        'Victoria': 'COAST',
        'Calhoun': 'COAST',
        'Jackson': 'COAST',
        'Warton': 'COAST',
        'Matagorda': 'COAST',
        'Fort Bend': 'COAST',
        'Brazoria': 'COAST',
        'Waller': 'COAST',
        'Harris': 'COAST',
        'Galveston': 'COAST',
        'Montgomery': 'COAST',
        'San Jacinto': 'COAST',
        'Liberty': 'COAST',
        'Chambers': 'COAST',
        'Polk': 'COAST',
        'Hardin': 'COAST',
        'Jefferson': 'COAST',
        'Tyler': 'COAST',
        'Jasper': 'COAST',
        'Orange': 'COAST',
        'Newton': 'COAST',
        'Wharton': 'COAST',
        'Hopkins': 'EAST',
        'Franklin': 'EAST',
        'Titus': 'EAST',
        'Morris': 'EAST',
        'Cass': 'EAST',
        'Rains': 'EAST',
        'Wood': 'EAST',
        'Marion': 'EAST',
        'Harrison': 'EAST',
        'Van Zandt': 'EAST',
        'Smith': 'EAST',
        'Gregg': 'EAST',
        'Henderson': 'EAST',
        'Rusk': 'EAST',
        'Panola': 'EAST',
        'Freestone': 'EAST',
        'Anderson': 'EAST',
        'Cherokee': 'EAST',
        'Nacogdoches': 'EAST',
        'Shelby': 'EAST',
        'San Augustine': 'EAST',
        'Sabine': 'EAST',
        'Robertson': 'EAST',
        'Leon': 'EAST',
        'Houston': 'EAST',
        'Angelina': 'EAST',
        'Trinity': 'EAST',
        'Brazos': 'EAST',
        'Madison': 'EAST',
        'Walker': 'EAST',
        'Grimes': 'EAST',
        'Camp': 'EAST',
        'Upshur': 'EAST',
        'El Paso': 'FAR_WEST',
        'Hudspeth': 'FAR_WEST',
        'Culberson': 'FAR_WEST',
        'Reeves': 'FAR_WEST',
        'Loving': 'FAR_WEST',
        'Winkler': 'FAR_WEST',
        'Ward': 'FAR_WEST',
        'Andrews': 'FAR_WEST',
        'Ector': 'FAR_WEST',
        'Crane': 'FAR_WEST',
        'Dawson': 'FAR_WEST',
        'Martin': 'FAR_WEST',
        'Midland': 'FAR_WEST',
        'Upton': 'FAR_WEST',
        'Borden': 'FAR_WEST',
        'Howard': 'FAR_WEST',
        'Glasscock': 'FAR_WEST',
        'Reagan': 'FAR_WEST',
        'Jeff Davis': 'FAR_WEST',
        'Pecos': 'FAR_WEST',
        'Crockett': 'FAR_WEST',
        'Presidio': 'FAR_WEST',
        'Brewster': 'FAR_WEST',
        'Terrell': 'FAR_WEST',
        'Cooke': 'NORTH',
        'Grayson': 'NORTH',
        'Fannin': 'NORTH',
        'Lamar': 'NORTH',
        'Red River': 'NORTH',
        'Bowie': 'NORTH',
        'Wheeler': 'NORTH',
        'Donley': 'NORTH',
        'Collingsworth': 'NORTH',
        'Brisco': 'NORTH',
        'Hall': 'NORTH',
        'Childress': 'NORTH',
        'Hardeman': 'NORTH',
        'Floyd': 'NORTH',
        'Motley': 'NORTH',
        'Cottle': 'NORTH',
        'Foard': 'NORTH',
        'Wilbarger': 'NORTH',
        'Wichita': 'NORTH',
        'Clay': 'NORTH',
        'Montague': 'NORTH',
        'Crosby': 'NORTH',
        'Dickens': 'NORTH',
        'King': 'NORTH',
        'Knox': 'NORTH',
        'Baylor': 'NORTH',
        'Archer': 'NORTH',
        'Garza': 'NORTH',
        'Kent': 'NORTH',
        'Stonewall': 'NORTH',
        'Haskell': 'NORTH',
        'Throckmorton': 'NORTH_CENTRAL',
        'Young': 'NORTH_CENTRAL',
        'Jack': 'NORTH_CENTRAL',
        'Wise': 'NORTH_CENTRAL',
        'Denton': 'NORTH_CENTRAL',
        'Collin': 'NORTH_CENTRAL',
        'Hunt': 'NORTH_CENTRAL',
        'Delta': 'NORTH_CENTRAL',
        'Shackelford': 'NORTH_CENTRAL',
        'Stephens': 'NORTH_CENTRAL',
        'Palo Pinto': 'NORTH_CENTRAL',
        'Parker': 'NORTH_CENTRAL',
        'Tarrant': 'NORTH_CENTRAL',
        'Dallas': 'NORTH_CENTRAL',
        'Rockwall': 'NORTH_CENTRAL',
        'Kaufman': 'NORTH_CENTRAL',
        'Callahan': 'NORTH_CENTRAL',
        'Eastland': 'NORTH_CENTRAL',
        'Erath': 'NORTH_CENTRAL',
        'Hood': 'NORTH_CENTRAL',
        'Somervell': 'NORTH_CENTRAL',
        'Johnson': 'NORTH_CENTRAL',
        'Ellis': 'NORTH_CENTRAL',
        'Brown': 'NORTH_CENTRAL',
        'Comanche': 'NORTH_CENTRAL',
        'Hamilton': 'NORTH_CENTRAL',
        'Bosque': 'NORTH_CENTRAL',
        'Hill': 'NORTH_CENTRAL',
        'Navarro': 'NORTH_CENTRAL',
        'Coryell': 'NORTH_CENTRAL',
        'McLennan': 'NORTH_CENTRAL',
        'Limestone': 'NORTH_CENTRAL',
        'Bell': 'NORTH_CENTRAL',
        'Falls': 'NORTH_CENTRAL',
        'Mills': 'NORTH_CENTRAL',
        'Austin': 'SOUTH_CENTRAL',
        'Bandera': 'SOUTH_CENTRAL',
        'Bastrop': 'SOUTH_CENTRAL',
        'Bexar': 'SOUTH_CENTRAL',
        'Blanco': 'SOUTH_CENTRAL',
        'Burleson': 'SOUTH_CENTRAL',
        'Burnet': 'SOUTH_CENTRAL',
        'Caldwell': 'SOUTH_CENTRAL',
        'Colorado': 'SOUTH_CENTRAL',
        'Comal': 'SOUTH_CENTRAL',
        'DeWitt': 'SOUTH_CENTRAL',
        'Fayette': 'SOUTH_CENTRAL',
        'Gonzales': 'SOUTH_CENTRAL',
        'Guadalupe': 'SOUTH_CENTRAL',
        'Hays': 'SOUTH_CENTRAL',
        'Karnes': 'SOUTH_CENTRAL',
        'Kendall': 'SOUTH_CENTRAL',
        'Lavaca': 'SOUTH_CENTRAL',
        'Lee': 'SOUTH_CENTRAL',
        'Medina': 'SOUTH_CENTRAL',
        'Milam': 'SOUTH_CENTRAL',
        'Travis': 'SOUTH_CENTRAL',
        'Washington': 'SOUTH_CENTRAL',
        'Williamson': 'SOUTH_CENTRAL',
        'Wilson': 'SOUTH_CENTRAL',
        'Aransas': 'SOUTHERN',
        'Atascosa': 'SOUTHERN',
        'Bee': 'SOUTHERN',
        'Brooks': 'SOUTHERN',
        'Cameron': 'SOUTHERN',
        'Dimmit': 'SOUTHERN',
        'Duval': 'SOUTHERN',
        'Frio': 'SOUTHERN',
        'Goliad': 'SOUTHERN',
        'Hidalgo': 'SOUTHERN',
        'Jim Hogg': 'SOUTHERN',
        'Jim Wells': 'SOUTHERN',
        'Kenedy': 'SOUTHERN',
        'Kleberg': 'SOUTHERN',
        'La Salle': 'SOUTHERN',
        'Live Oak': 'SOUTHERN',
        'Maverick': 'SOUTHERN',
        'McMullen': 'SOUTHERN',
        'Nueces': 'SOUTHERN',
        'Refugio': 'SOUTHERN',
        'San Patricio': 'SOUTHERN',
        'Starr': 'SOUTHERN',
        'Webb': 'SOUTHERN',
        'Willacy': 'SOUTHERN',
        'Zapata': 'SOUTHERN',
        'Zavala': 'SOUTHERN',
        'Coke': 'WEST',
        'Coleman': 'WEST',
        'Concho': 'WEST',
        'Edwards': 'WEST',
        'Fisher': 'WEST',
        'Gillespie': 'WEST',
        'Irion': 'WEST',
        'Jones': 'WEST',
        'Kerr': 'WEST',
        'Kimble': 'WEST',
        'Kinney': 'WEST',
        'Lampasas': 'WEST',
        'Llano': 'WEST',
        'Mason': 'WEST',
        'McCulloch': 'WEST',
        'Menard': 'WEST',
        'Mitchell': 'WEST',
        'Nolan': 'WEST',
        'Real': 'WEST',
        'Runnels': 'WEST',
        'San Saba': 'WEST',
        'Schleicher': 'WEST',
        'Scurry': 'WEST',
        'Sterling': 'WEST',
        'Sutton': 'WEST',
        'Taylor': 'WEST',
        'Tom Green': 'WEST',
        'Uvalde': 'WEST',
        'Val Verde': 'WEST'
    }



    # 将映射字典转换为一个DataFrame，方便后续合并
    mapping_df = pd.DataFrame(list(erlot_region_mapping.items()), columns=['NAME', 'ERCOT_Region'])

    # 3. 将ERCOT区域信息合并到德州地理数据中
    # 使用左连接，确保所有德州县都被保留，即使有些县不在您的映射里（这些县区域会为NaN）
    tx_counties_with_region = tx_counties.merge(mapping_df, on='NAME', how='left')
    regions_gdf = tx_counties_with_region.dissolve(by='ERCOT_Region')

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小


    colors = []
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
        colors.append(color)

        print(i)
        #guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        #guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省


    region_colors = {
        'COAST': colors[0],
        'EAST': colors[1],
        'FAR_WEST': colors[2],
        'NORTH': colors[3],
        'NORTH_CENTRAL': colors[4],
        'SOUTH_CENTRAL': colors[5],
        'SOUTHERN': colors[6],
        'WEST': colors[7]
        # 为其他区域添加颜色...
    }
    colors = [region_colors.get(region, 'gray') for region in regions_gdf.index]

    regions_gdf.plot(ax=ax,
                     color=colors,
                     edgecolor='white',
                     linewidth=1.5,
                     alpha=0.7)

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
    #ax_inset_2.yaxis.set_tick_params(labelsize=ticksize)

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


#plot_texas_map()


def plot_pjm():
    import geopandas as gpd
    counties = gpd.read_file("map_data/tl_2021_us_county.zip")
    print(counties)


#plot_pjm()


def plot_India_map(titlesize=16, ticksize=14, labelsize=14, region='india'):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path



    city_name = ['Maharashtra_data_2017_2023',
                    'Delhi_data_2017_2023']

    cities = ['Maharashtra', 'Delhi']
    colors = []

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)
    # 筛选出德州数据
    states = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip")

    # 筛选出Maharashtra和Delhi
    target_regions = states[states['name'].isin(['Maharashtra', 'Delhi'])]



    cmap = plt.get_cmap('Blues')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(0, 0.5)

    #india = world[world['NAME'] == 'India']  # 确保为WGS84坐标系
    #india.plot(ax=ax, color=cmap(norm(1.2)), edgecolor='white', alpha=0.7, linewidth=1)


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

        strat_time = '2018/01/01/00'
        end_time = '2019/04/30/23'
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
        T_95 = np.percentile(T_i_list, 90)

        # load and temperature slices formulation
        coldwave_index = []
        hotwave_index = []
        common_load_mean = []
        hotwave_load_mean = []
        coldwave_load_mean = []

        for j in range(30, load.shape[0] // 24 - 3 - 30 - 6):

            ## define the cold wave index
            ECI_sig = np.mean(T_i_list[j:j + 1]) - T_05
            ECI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            ECF = min(0, -ECI_sig * min(-1, ECI_accl))
            coldwave_index.append(float(ECF < 0))
            if ECF < 0:
                coldwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                coldwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])

            ## define the hot wave index
            EHI_sig = np.mean(T_i_list[j:j + 1]) - T_95
            EHI_accl = np.mean(T_i_list[j:j + 3]) - np.mean(T_i_list[j - 30:j])
            EHF = max(0, EHI_sig * max(1, EHI_accl))
            hotwave_index.append(float(EHF > 0))

            if EHF > 0:
                hotwave_load_mean.append(load[24 * j: 24 * (j + 1)])
                hotwave_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                hotwave_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])

            else:
                common_load_mean.append(load[24 * j: 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])
                common_tem_norm.append(norm_tem[24 * j: 24 * (j + 1)])


        hot_common_ratio = np.mean(hotwave_load_norm)/np.mean(common_load_norm)-1
        color = cmap(norm(hot_common_ratio))
        colors.append(color)

        print(hot_common_ratio)
        print(i)
        #guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        #guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    target_regions.plot(ax=ax, color=colors,
                        edgecolor='white',
                        linewidth=1.5,
                        alpha=0.7)

    # 突出显示 Delhi 位置（使用点标记）
    delhi_coords = (77.1025, 28.7041)  # Delhi 的经纬度
    ax.scatter(delhi_coords[0], delhi_coords[1], color='red', s=200, zorder=5,
               marker='*', edgecolor='darkred', linewidth=2)

    # 添加标注
    ax.annotate('Delhi', xy=delhi_coords, xytext=(77.5, 29.0),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))


    ax_inset_zoom = inset_axes(
        ax,
        width="100%", height="100%",
        # loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.65, 0.65, 0.3, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )
    delhi_region = states[states['name'].str.contains('Delhi', na=False)]
    delhi_region.plot(ax=ax_inset_zoom, color=colors[1], alpha=0.8, edgecolor=colors[1], linewidth=2)
    ax_inset_zoom.set_title('Delhi (Enlarged)', fontsize=titlesize-2)
    ax_inset_zoom.set_xticks([77.125])  # 东经
    ax_inset_zoom.set_yticks([28.5, 28.75])  # 北纬
    ax_inset_zoom.set_xticklabels(['77.125°E'])
    ax_inset_zoom.set_yticklabels(['28.5°N', '28.75°N'])
    ax_inset_zoom.set_box_aspect(0.6)
    ax_inset_zoom.xaxis.set_tick_params(labelsize=ticksize-4)
    ax_inset_zoom.yaxis.set_tick_params(labelsize=ticksize-4)
    #ax_inset_zoom.set_axis_off()

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

    # 印度的典型经纬度范围
    ax.set_xticks([68, 76, 84, 92, 100])  # 东经
    ax.set_yticks([5, 13, 21, 29])  # 北纬

    ax.set_xticklabels(['68°E', '76°E', '84°E', '92°E', '100°E'])
    ax.set_yticklabels(['5°N', '13°N', '21°N', '29°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Patterns', fontsize=labelsize)
    ax_inset.set_xlabel('Hour', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_xlim(68, 97)  # 德州经度范围
    ax.set_ylim(5, 34)  # 德州纬度范围
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(0.8)

    # 设置标题和显示
    ax.set_title("Heatwave in India", fontsize=titlesize)
    plt.tight_layout()
    #plt.axis('equal')
    fig.savefig('figures/load_change_in_{}.pdf'.format(region))
    plt.show()

#plot_India_map()

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
                'Pennsylvania Electric Company',
                'Pennsylvania Power and Light Company',
                'Potomac Electric Power',
                'Public Service Electric and Gas Company']

    pjm_states = [
        'Pennsylvania', 'New Jersey', 'Maryland', 'Delaware',
        'Virginia', 'West Virginia', 'Ohio', 'Kentucky',
        'Illinois', 'Indiana', 'Michigan', 'North Carolina', 'Tennessee'
    ]

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
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')

    world_projected.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)


    colors=[]

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
        colors.append(color)

        print(i)

    for i in range(len(pjm_zones)):
        zone_name = pjm_zones[i]
        color = colors[i]  # 使用取模确保颜色循环使用

        # 筛选当前区域
        current_zone = pjm_wgs84[pjm_wgs84['Zone_Name'] == zone_name]  # 根据实际列名调整


        current_zone.plot(ax=ax, color=color, alpha=0.7,
                              edgecolor='white', linewidth=1.5)


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

#plot_PJM_map()