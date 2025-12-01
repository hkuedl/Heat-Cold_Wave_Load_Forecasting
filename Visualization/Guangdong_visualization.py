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
import requests
from io import BytesIO
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import torch

area = 'HY'


def plot_scatter(data_name = '欧洲2018寒潮'):

    data=pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（{}）'.format(data_name),
                       header=1, usecols='B:Q').values
    #print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    plt.figure(figsize=(4, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    for i in range(1, 7):
        x = data[:, 0]
        y = data[:, i*2-1]
        plt.scatter(x, y, alpha=0.6, edgecolor=(0.1, 0.2, 0.8), facecolors='none')



    plt.plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    plt.annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=(0.1, 0.2, 0.8), arrowstyle='->', lw=1.5))
    plt.annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color='orange', arrowstyle='->', lw=1.5))
    plt.xlabel('Proposed', fontdict=text_font)
    plt.ylabel('Baselines', fontdict=text_font)
    plt.xticks(np.arange(0, 0.125, 0.02), fontsize=12)  # x轴每隔0.25显示一个点，字体大小为12
    plt.yticks(np.arange(0, 0.125, 0.02), fontsize=12)  # y轴每隔0.25显示一个点，字体大小为12
    plt.xlim(0, 0.1)
    plt.ylim(0, 0.1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


def plot_scatter_subplot_guangdong(data_name = '广东2023热浪', region='guangdong'):

    color1 = '#46AEA0FF'
    color2 = '#F8A070FF'


    data=pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（{}）'.format(data_name),
                       header=1, usecols='B:Q').values
    #print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))


    #plt.figure(figsize=(4, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    for i in range(1, 7):
        x = data[:, 0]
        y = data[:, i*2]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

        #axs[0].scatter(x, y, alpha=0.6, edgecolor=(0.1, 0.2, 0.8), facecolors='none')



    axs[0].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[0].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[0].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[0].set_xlabel('Proposed', fontdict=text_font)
    axs[0].set_ylabel('Baselines', fontdict=text_font)
    axs[0].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[0].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[0].set_xlim(0, 0.1)
    axs[0].set_ylim(0, 0.1)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('nMAE', fontsize=15)
    #axs[0].grid()
    axs[0].tick_params(axis='both', labelsize=12)




    for i in range(1, 7):
        x = data[:, 1]
        y = data[:, i*2+1]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

    axs[1].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[1].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[1].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[1].set_xlabel('Proposed', fontdict=text_font)
    axs[1].set_ylabel('Baselines', fontdict=text_font)
    axs[1].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[1].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[1].set_xlim(0, 0.1)
    axs[1].set_ylim(0, 0.1)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title('nRMSE', fontsize=15)
    fig.suptitle('2022 Guangdong Heatwave', fontsize=15)
    #axs[0].grid()
    axs[1].tick_params(axis='both', labelsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig('figures/scatter_{}.eps'.format(region))
    plt.show()

def plot_scatter_subplot_hunan(data_name = '湖南2023寒潮', region='hunan'):
    color1 = '#46AEA0FF'
    color2 = '#F8A070FF'

    data=pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（{}）'.format(data_name),
                       header=1, usecols='B:Q').values
    #print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))


    #plt.figure(figsize=(4, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    for i in range(1, 7):
        x = data[:, 0]
        y = data[:, i*2]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

        #axs[0].scatter(x, y, alpha=0.6, edgecolor=(0.1, 0.2, 0.8), facecolors='none')



    axs[0].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[0].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[0].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[0].set_xlabel('Proposed', fontdict=text_font)
    axs[0].set_ylabel('Baselines', fontdict=text_font)
    axs[0].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[0].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[0].set_xlim(0, 0.1)
    axs[0].set_ylim(0, 0.1)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('nMAE', fontsize=15)
    #axs[0].grid()
    axs[0].tick_params(axis='both', labelsize=12)




    for i in range(1, 7):
        x = data[:, 1]
        y = data[:, i*2+1]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

    axs[1].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[1].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[1].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[1].set_xlabel('Proposed', fontdict=text_font)
    axs[1].set_ylabel('Baselines', fontdict=text_font)
    axs[1].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[1].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[1].set_xlim(0, 0.1)
    axs[1].set_ylim(0, 0.1)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title('nRMSE', fontsize=15)
    fig.suptitle('2023 Hunan Coldwave', fontsize=15)
    #axs[0].grid()
    axs[1].tick_params(axis='both', labelsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig('figures/scatter_{}.eps'.format(region))
    plt.show()

def plot_scatter_subplot_PJM(data_name = 'PJM2023热浪', region='pjm'):
    color1 = '#46AEA0FF'
    color2 = '#F8A070FF'

    data=pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（{}）'.format(data_name),
                       header=1, usecols='B:Q').values
    #print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))


    #plt.figure(figsize=(4, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    for i in range(1, 7):
        x = data[:, 0]
        y = data[:, i*2]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

        #axs[0].scatter(x, y, alpha=0.6, edgecolor=(0.1, 0.2, 0.8), facecolors='none')



    axs[0].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[0].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[0].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[0].set_xlabel('Proposed', fontdict=text_font)
    axs[0].set_ylabel('Baselines', fontdict=text_font)
    axs[0].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[0].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[0].set_xlim(0, 0.1)
    axs[0].set_ylim(0, 0.1)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('nMAE', fontsize=15)
    #axs[0].grid()
    axs[0].tick_params(axis='both', labelsize=12)




    for i in range(1, 7):
        x = data[:, 1]
        y = data[:, i*2+1]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

    axs[1].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[1].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[1].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[1].set_xlabel('Proposed', fontdict=text_font)
    axs[1].set_ylabel('Baselines', fontdict=text_font)
    axs[1].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[1].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[1].set_xlim(0, 0.1)
    axs[1].set_ylim(0, 0.1)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title('nRMSE', fontsize=15)
    fig.suptitle('2023 PJM Heatwaves', fontsize=15)
    #axs[0].grid()
    axs[1].tick_params(axis='both', labelsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig('figures/scatter_{}.eps'.format(region))
    plt.show()

def plot_scatter_subplot_Texas(data_name = '德州2023热浪', region='texas'):
    color1 = '#46AEA0FF'
    color2 = '#F8A070FF'

    data=pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（{}）'.format(data_name),
                       header=1, usecols='B:Q').values
    #print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))


    #plt.figure(figsize=(4, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    for i in range(1, 7):
        x = data[:, 0]
        y = data[:, i*2]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

        #axs[0].scatter(x, y, alpha=0.6, edgecolor=(0.1, 0.2, 0.8), facecolors='none')



    axs[0].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[0].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[0].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[0].set_xlabel('Proposed', fontdict=text_font)
    axs[0].set_ylabel('Baselines', fontdict=text_font)
    axs[0].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[0].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[0].set_xlim(0, 0.1)
    axs[0].set_ylim(0, 0.1)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('nMAE', fontsize=15)
    #axs[0].grid()
    axs[0].tick_params(axis='both', labelsize=12)




    for i in range(1, 7):
        x = data[:, 1]
        y = data[:, i*2+1]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

    axs[1].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[1].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[1].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[1].set_xlabel('Proposed', fontdict=text_font)
    axs[1].set_ylabel('Baselines', fontdict=text_font)
    axs[1].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[1].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[1].set_xlim(0, 0.1)
    axs[1].set_ylim(0, 0.1)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title('nRMSE', fontsize=15)
    fig.suptitle('2023 Texas Heatwave', fontsize=15)
    #axs[0].grid()
    axs[1].tick_params(axis='both', labelsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig('figures/scatter_{}.eps'.format(region))
    plt.show()

def plot_scatter_subplot_India(data_name = '印度2022热浪', region='india'):
    color1 = '#46AEA0FF'
    color2 = '#F8A070FF'

    data=pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（{}）'.format(data_name),
                       header=1, usecols='B:Q').values
    #print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))


    #plt.figure(figsize=(4, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    for i in range(1, 7):
        x = data[:, 0]
        y = data[:, i*2]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

        #axs[0].scatter(x, y, alpha=0.6, edgecolor=(0.1, 0.2, 0.8), facecolors='none')



    axs[0].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[0].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[0].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[0].set_xlabel('Proposed', fontdict=text_font)
    axs[0].set_ylabel('Baselines', fontdict=text_font)
    axs[0].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[0].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[0].set_xlim(0, 0.1)
    axs[0].set_ylim(0, 0.1)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('nMAE', fontsize=15)
    #axs[0].grid()
    axs[0].tick_params(axis='both', labelsize=12)




    for i in range(1, 8):
        x = data[:, 1]
        y = data[:, i*2+1]
        for j in range(x.shape[0]-1):
            if x[j] <= y[j]:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

    axs[1].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[1].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[1].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[1].set_xlabel('Proposed', fontdict=text_font)
    axs[1].set_ylabel('Baselines', fontdict=text_font)
    axs[1].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[1].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[1].set_xlim(0, 0.1)
    axs[1].set_ylim(0, 0.1)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title('nRMSE', fontsize=15)
    fig.suptitle('2022 India Heatwave', fontsize=15)
    #axs[0].grid()
    axs[1].tick_params(axis='both', labelsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig('figures/scatter_{}.eps'.format(region))
    plt.show()

def plot_scatter_subplot_Europe(data_name = '欧洲2018寒潮', region='europe'):
    color1 = '#46AEA0FF'
    color2 = '#F8A070FF'

    data=pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（{}）'.format(data_name),
                       header=1, usecols='B:Q').values
    #print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))


    #plt.figure(figsize=(4, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    for i in range(1, 7):
        x = data[:, 0]
        y = data[:, i*2]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[0].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

        #axs[0].scatter(x, y, alpha=0.6, edgecolor=(0.1, 0.2, 0.8), facecolors='none')



    axs[0].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[0].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[0].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[0].set_xlabel('Proposed', fontdict=text_font)
    axs[0].set_ylabel('Baselines', fontdict=text_font)
    axs[0].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[0].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[0].set_xlim(0, 0.1)
    axs[0].set_ylim(0, 0.1)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('nMAE', fontsize=15)
    #axs[0].grid()
    axs[0].tick_params(axis='both', labelsize=12)




    for i in range(1, 7):
        x = data[:, 1]
        y = data[:, i*2+1]
        for j in range(x.shape[0]):
            if x[j] <= y[j]:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color1, facecolors='none')
            else:
                axs[1].scatter(x[j], y[j], alpha=0.6, edgecolor=color2, facecolors='none')

    axs[1].plot([0, 0.095], [0, 0.095], 'k--', lw=1)
    axs[1].annotate('Proposed\nbetter', xy=(0.00, 0.1), xytext=(0.01, 0.07),
                 fontsize = 15,
                 arrowprops=dict(color=color1, arrowstyle='->', lw=1.5))
    axs[1].annotate('Baselines\nbetter', xy=(0.1, 0.0), xytext=(0.055, 0.025),
                 fontsize = 15,
                 arrowprops=dict(color=color2, arrowstyle='->', lw=1.5))
    axs[1].set_xlabel('Proposed', fontdict=text_font)
    axs[1].set_ylabel('Baselines', fontdict=text_font)
    axs[1].set_xticks(np.arange(0, 0.125, 0.02))  # x轴每隔0.25显示一个点，字体大小为12
    axs[1].set_yticks(np.arange(0, 0.125, 0.02))  # y轴每隔0.25显示一个点，字体大小为12
    axs[1].set_xlim(0, 0.1)
    axs[1].set_ylim(0, 0.1)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_title('nRMSE', fontsize=15)
    fig.suptitle('2018 Europe Coldwave', fontsize=15)
    #axs[0].grid()
    axs[1].tick_params(axis='both', labelsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig('figures/scatter_{}.eps'.format(region))
    plt.show()

#plot_scatter()
#plot_scatter_subplot_guangdong()
#plot_scatter_subplot_PJM()
#plot_scatter_subplot_Texas()
#plot_scatter_subplot_India()
#plot_scatter_subplot_hunan()
#plot_scatter_subplot_Europe()

def plot_bar_subplot_nMAE(data_name = '广东2023热浪'):
    data_1 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（广东2023热浪）'.format(data_name),
                         header=1, usecols='B:Q').values
    data_2 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（PJM2023热浪）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_3 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（德州2023热浪）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_4 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（印度2022热浪）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_5 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（湖南2023寒潮）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_6 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（欧洲2018寒潮）'.format(data_name),
                           header=1, usecols='B:Q').values

    # print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    #fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


    # Build a dataset
    df = pd.DataFrame({
        "name": ['Proposed', 'MLP', 'LSTM', 'CNN', 'NBEATS', 'ImpactNet', 'Informer', 'Autoformer']*6,
        "value": [data_1[-1, 2*i]-0.02 for i in range(8)] +
                 [data_2[-1, 2*i]-0.02 for i in range(8)] +
                 [data_3[-1, 2*i]-0.02 for i in range(8)] +
                 [data_4[-1, 2*i]-0.02 for i in range(8)] +
                 [data_5[-1, 2*i]-0.02 for i in range(8)] +
                 [data_6[-1, 2*i]-0.02 for i in range(8)],
        "group": ["Guangdong"] * 8 + ["PJM"] * 8 + ["Texas"] * 8 + ["India"] * 8
                 + ["Hunan"] * 8 + ["Europe"] * 8
    })

    # Show 3 first rows
    print(df.head(3))

    def get_label_rotation(angle, offset):
        # Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle + offset)
        if angle <= np.pi:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"
        return rotation, alignment



    def add_labels(angles, values, labels, offset, ax):

        # This is the space between the end of the bar and the label
        padding = 0.001

        # Iterate over angles, values, and labels, to add all of them.
        for angle, value, label, in zip(angles, values, labels):
            angle = angle

            # Obtain text rotation and alignment
            rotation, alignment = get_label_rotation(angle, offset)

            # And finally add the text
            if label == 'Proposed':
                ax.text(
                    x=angle,
                    y=value + padding,
                    s=label,
                    ha=alignment,
                    va="center",
                    rotation=rotation,
                    rotation_mode="anchor",
                    fontweight="bold",
                    fontsize=11
                )
            else:
                ax.text(
                    x=angle,
                    y=value + padding,
                    s=label,
                    ha=alignment,
                    va="center",
                    rotation=rotation,
                    rotation_mode="anchor",
                    fontsize=11
                )


    # Determines where to place the first bar.
    # By default, matplotlib starts at 0 (the first bar is horizontal)
    # but here we say we want to start at pi/2 (90 deg)
    OFFSET = np.pi / 2

    # Reorder the dataframe
    df_sorted = (
        df
        .groupby(["group"])
        .apply(lambda x: x.sort_values(["value"], ascending=False))
        .reset_index(drop=True)
    )

    # All this part is like the code above
    VALUES = df_sorted["value"].values
    LABELS = df_sorted["name"].values
    GROUP = df_sorted["group"].values

    PAD = 3
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]

    offset = 0
    IDXS = []
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-0.08, 0.06)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]
    COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]
    COLORS = ['#1D4F60FF']
    COLORS = ['#46AEA0FF'] * 8 + ['#98D048FF'] * 8 + ['#F8D068FF'] * 8 + ['#88A0DCFF'] * 8 + \
             ['#F6B8BDFF'] * 8 + ['#F8A070FF'] * 8

    ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
        edgecolor="white", linewidth=2
    )

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

    # Extra customization below here --------------------

    # This iterates over the sizes of the groups adding reference
    # lines and annotations.

    offset = 0
    for group, size in zip(["GD\nnMAE", "PJM\nnMAE",
                            "Texas\nnMAE", "India\nnMAE",
                            "HN\nnMAE", "Europe\nnMAE"], GROUPS_SIZE):
        # Add line below bars
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-0.005] * 50, color="#333333")

        # Add text to indicate group
        ax.text(
            np.mean(x1), -0.03, group, color="#333333", fontsize=12,
            fontweight="bold", ha="center", va="center"
        )

        # Add reference lines at 20, 40, 60, and 80
        x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
        ax.plot(x2, [0.02] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [0.04] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [0.06] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [0.08] * 50, color="#bebebe", lw=0.8)

        offset += size + PAD

    plt.savefig('figures/rose_nmae.eps')
    plt.tight_layout()
    plt.show()


def plot_bar_subplot_nRMSE(data_name = '广东2023热浪'):
    data_1 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（广东2023热浪）'.format(data_name),
                         header=1, usecols='B:Q').values
    data_2 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（PJM2023热浪）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_3 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（德州2023热浪）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_4 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（印度2022热浪）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_5 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（湖南2023寒潮）'.format(data_name),
                           header=1, usecols='B:Q').values
    data_6 = pd.read_excel('../极端温度实验结果.xlsx', sheet_name='基础（欧洲2018寒潮）'.format(data_name),
                           header=1, usecols='B:Q').values

    # print(data.values)
    text_font = {'size': 15}
    num_font = {'size': 12}

    #fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


    # Build a dataset
    df = pd.DataFrame({
        "name": ['Proposed', 'MLP', 'LSTM', 'CNN', 'NBEATS', 'ImpactNet', 'Informer', 'Autoformer']*6,
        "value": [data_1[-1, 2*i+1]-0.02 for i in range(8)] +
                 [data_2[-1, 2*i+1]-0.02 for i in range(8)] +
                 [data_3[-1, 2*i+1]-0.02 for i in range(8)] +
                 [data_4[-1, 2*i+1]-0.02 for i in range(8)] +
                 [data_5[-1, 2*i+1]-0.02 for i in range(8)] +
                 [data_6[-1, 2*i+1]-0.02 for i in range(8)],
        "group": ["Guangdong"] * 8 + ["PJM"] * 8 + ["Texas"] * 8 + ["India"] * 8
                 + ["Hunan"] * 8 + ["Europe"] * 8
    })

    # Show 3 first rows
    print(df.head(3))

    def get_label_rotation(angle, offset):
        # Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle + offset)
        if angle <= np.pi:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"
        return rotation, alignment



    def add_labels(angles, values, labels, offset, ax):

        # This is the space between the end of the bar and the label
        padding = 0.001

        # Iterate over angles, values, and labels, to add all of them.
        for angle, value, label, in zip(angles, values, labels):
            angle = angle

            # Obtain text rotation and alignment
            rotation, alignment = get_label_rotation(angle, offset)

            # And finally add the text
            if label == 'Proposed':
                ax.text(
                    x=angle,
                    y=value + padding,
                    s=label,
                    ha=alignment,
                    va="center",
                    rotation=rotation,
                    rotation_mode="anchor",
                    fontweight="bold",
                    fontsize=11
                )
            else:
                ax.text(
                    x=angle,
                    y=value + padding,
                    s=label,
                    ha=alignment,
                    va="center",
                    rotation=rotation,
                    rotation_mode="anchor",
                    fontsize=11
                )


    # Determines where to place the first bar.
    # By default, matplotlib starts at 0 (the first bar is horizontal)
    # but here we say we want to start at pi/2 (90 deg)
    OFFSET = np.pi / 2

    # Reorder the dataframe
    df_sorted = (
        df
        .groupby(["group"])
        .apply(lambda x: x.sort_values(["value"], ascending=False))
        .reset_index(drop=True)
    )

    # All this part is like the code above
    VALUES = df_sorted["value"].values
    LABELS = df_sorted["name"].values
    GROUP = df_sorted["group"].values

    PAD = 3
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]

    offset = 0
    IDXS = []
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-0.08, 0.06)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]
    COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]
    COLORS = ['#1D4F60FF']
    COLORS = ['#46AEA0FF'] * 8 + ['#98D048FF'] * 8 + ['#F8D068FF'] * 8 + ['#88A0DCFF'] * 8 + \
             ['#F6B8BDFF'] * 8 + ['#F8A070FF'] * 8

    ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
        edgecolor="white", linewidth=2
    )

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

    # Extra customization below here --------------------

    # This iterates over the sizes of the groups adding reference
    # lines and annotations.

    offset = 0
    for group, size in zip(["GD\nnRMSE", "PJM\nnRMSE",
                            "Texas\nnRMSE", "India\nnRMSE",
                            "HN\nnRMSE", "Europe\nnRMSE"], GROUPS_SIZE):
        # Add line below bars
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-0.005] * 50, color="#333333")

        # Add text to indicate group
        ax.text(
            np.mean(x1), -0.03, group, color="#333333", fontsize=12,
            fontweight="bold", ha="center", va="center"
        )

        # Add reference lines at 20, 40, 60, and 80
        x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
        ax.plot(x2, [0.02] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [0.04] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [0.06] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [0.08] * 50, color="#bebebe", lw=0.8)
        #ax.text(0, -0.1, 'Center Text', va='center', fontsize=12, color='black')

        #plt.title('Per Dataset nRMSE Comparison of All Methods')

        offset += size + PAD

    plt.savefig('figures/rose_nrmse.eps')
    plt.tight_layout()
    plt.show()



#plot_bar_subplot_nMAE()
#plot_bar_subplot_nRMSE()





def plot_rosy():
    # 示例数据
    categories = ['A', 'B', 'C', 'D', 'E', 'F']
    values = [4, 7, 1, 8, 5, 6]

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

    # 将数据封闭成一个循环
    values += values[:1]
    angles += angles[:1]

    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 绘制南丁格尔玫瑰图
    ax.fill(angles, values, color='lightblue', alpha=0.6)
    ax.plot(angles, values, color='blue', linewidth=2)

    # 设置标题和标签
    ax.set_title('Nightingale Rose Diagram', loc='center', pad=20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # 显示图形
    plt.show()



def plot_box():
    # 示例数据
    categories = ['Cold Surge', 'Sandstorm', 'Typhoon', 'Gale', 'Heatwave', 'Rainstorm', 'Blizzard']
    np.random.seed(0)

    # 生成示例数据
    data_north = [np.random.normal(0, std, 50) for std in range(1, 8)]
    data_south = [np.random.normal(0, std + 1, 100) for std in range(1, 8)]

    # 创建图形
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # 绘制北方的小提琴图
    sns.violinplot(data=data_north, ax=ax[0], inner=None, color='lightgray', linewidth=1.5)
    sns.violinplot(data=data_north, ax=ax[0], inner='quartile', linewidth=1.5)
    sns.violinplot(data=data_south, ax=ax[0], inner=None, color='lightgray', linewidth=1.5, alpha=0.5)

    ax[0].set_title('North', fontsize=16)
    ax[0].set_xticklabels(categories)

    # 绘制南方的小提琴图
    sns.violinplot(data=data_south, ax=ax[1], inner=None, color='lightgray', linewidth=1.5)
    sns.violinplot(data=data_south, ax=ax[1], inner='quartile', linewidth=1.5)
    ax[1].set_title('South', fontsize=16)
    ax[1].set_xticklabels(categories)

    # 添加参考线
    for a in ax:
        a.axhline(0, color='red', linestyle='--', linewidth=1)  # 添加虚线
        a.set_ylabel('Value', fontsize=14)

    # 调整布局
    plt.tight_layout()
    plt.show()


#plot_box()
#plot_rosy()


def extreme_visualization():
    country = 'Belgium'
    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
    start_date = pd.to_datetime('2015/01/01/00')
    end_date = pd.to_datetime('2019/4/30/23')
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    load = np.array(data['Load'])
    temperature = np.array(data['Temperature'])

    T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                          np.min(temperature[24 * i:24 * (i + 1)])) / 2
                         for i in range(temperature.shape[0] // 24)])

    #print(T_05)
    # define the coldwave index
    T_05 = np.percentile(T_i_list, 5)
    ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                        for i in range(T_i_list.shape[0] - 3 - 30)])
    ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                         for i in range(T_i_list.shape[0] - 3 - 30)])
    ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

    # define the heatwave index
    T_95 = np.percentile(T_i_list, 95)
    EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                        for i in range(T_i_list.shape[0] - 3 - 30)])
    EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                         for i in range(T_i_list.shape[0] - 3 - 30)])
    EHF = np.array([max(0, EHI_sig[i] * max(-1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])


    ramping_list=[]




def extreme_distribution():
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    country = 'Belgium'
    data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))
    start_date = pd.to_datetime('2015/01/01/00')
    end_date = pd.to_datetime('2019/4/30/23')
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    load = np.array(data['Load'])
    temperature = np.array(data['Temperature'])

    T_i_list = np.array([np.mean(temperature[24 * i:24 * (i + 1)])
                          for i in range(temperature.shape[0] // 24)])
    load_i_list = np.array([np.max(load[24 * i:24 * (i + 1)])
                          for i in range(temperature.shape[0] // 24)])

    T_05 = np.percentile(T_i_list, 5)

    tem_list = []
    load_list = []
    for i in range(10):

        percentile_tem_list = []
        percentile_load_list = []
        for j in range(len(T_i_list)):
            if T_i_list[j] >= (max(T_i_list)-min(T_i_list))*i/10+min(T_i_list) and \
                T_i_list[j] < (max(T_i_list)-min(T_i_list))*(i+1)/10+min(T_i_list):
                percentile_tem_list.append(T_i_list[j])
                percentile_load_list.append(load_i_list[j])

        tem_list.append(percentile_tem_list)
        load_list.append(percentile_load_list)

    # we define a dictionnary with months that we'll use later
    month_dict = [str((max(T_i_list)-min(T_i_list))*i/10+min(T_i_list)) for i in range(11)]

    df = pd.DataFrame({
        "tem": list(itertools.chain.from_iterable(tem_list)),
        "load": list(itertools.chain.from_iterable(load_list)),
        "group": ['1'] * len(tem_list[0]) + ['2'] * len(tem_list[1]) +
                 ['3'] * len(tem_list[2]) + ['4'] * len(tem_list[3]) +
                 ['5'] * len(tem_list[4]) + ['6'] * len(tem_list[5]) +
                 ['7'] * len(tem_list[6]) + ['8'] * len(tem_list[7]) +
                 ['9'] * len(tem_list[8]) + ['10'] * len(tem_list[9])

    })

    month_mean_serie = df.groupby('group')['load'].mean()
    df['mean_load'] = df['group'].map(month_mean_serie)

    # 计算每个 group 的样本比例
    group_counts = df['group'].value_counts(normalize=True)

    # 创建 FacetGrid
    pal = sns.color_palette(palette='coolwarm', n_colors=10)
    g = sns.FacetGrid(df, row='group', hue='mean_load', aspect=15, height=0.75, palette=pal)

    # 定义一个绘制调整后的 KDE 的函数
    def weighted_kdeplot(data, group_name, **kwargs):
        #print(data.head(3))  # 打印传入的 Series 数据
        # 获取当前组的比例
        proportion = group_counts[group_name]

        # 计算 KDE
        kde = sns.kdeplot(data, **kwargs)  # 使用传入的 Series 数据
        # 将密度值乘以比例
        for line in kde.lines:
            line.set_ydata(line.get_ydata() * proportion)

    # 在 FacetGrid 中绘制调整后的 KDE
    for group in df['group'].unique():
        g.map(weighted_kdeplot, 'load', bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5, group_name=group)

    # 添加轮廓线
    for group in df['group'].unique():
        g.map(weighted_kdeplot, 'load', bw_adjust=1, clip_on=False, color="w", lw=2, group_name=group)


    # 添加文本标签
    for i, ax in enumerate(g.axes.flat):
        ax.text(7000, 0, month_dict[i + 1], fontweight='bold', fontsize=15, color=ax.lines[-1].get_color())

    # 调整子图
    g.fig.subplots_adjust(hspace=-0.3)

    # 清除轴标题和刻度
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    # 设置 x 轴标签和标题
    #plt.setp(g.axes, xticks=range(0, 10000, 1000), fontsize=15, fontweight='bold')
    #plt.xlabel('Temperature in degree Celsius', fontweight='bold', fontsize=15)
    #g.fig.suptitle('Daily average temperature in Belgium per month', ha='right', fontsize=20, fontweight=20)

    plt.show()



def plotArrow3D(ax, x, y, z, dx, dy, dz, arrowLen=1.0, color=None, mutation_scale=10, lw=2):
    """`ax` is a matplotlib `Axes3D` object.
        `x`, `y`, `z` is the start point of the arrow.
        `dx`, `dy`, `dz` is the vector of the arrow.
        `arrowLen` is the length of the arrow head.
        `color` is the color of the arrow.
        `mutation_scale` is the scale factor of the arrow head.
        `lw` is the line width of the arrow.
    """
    # 计算箭头的长度和角度
    length = arrowLen * np.sqrt(dx**2 + dy**2 + dz**2)
    alpha = np.arccos(dz/length)
    beta = np.arctan2(dy,dx)

    # 绘制箭头线
    ax.plot([x, x+dx], [y, y+dy], [z, z+dz], color=color, lw=lw)

    # 绘制箭头尖端
    ax.plot([x+dx], [y+dy], [z+dz], marker='x', color=color, markersize=8)

    # 绘制箭头头部
    X,Y,Z = arrowHead(x+dx, y+dy, z+dz, alpha, beta, length, mutation_scale)
    ax.plot(X, Y, Z, color=color, lw=lw)

def arrowHead(x, y, z, alpha, beta, length, mutation_scale):
    """计算箭头头部坐标"""
    X,Y,Z = [],[],[]

    # 计算箭头尖端的坐标
    X.append(x)
    Y.append(y)
    Z.append(z)

    X.append(x - 0.5*mutation_scale*length*np.sin(alpha)*np.cos(beta))
    Y.append(y - 0.5*mutation_scale*length*np.sin(alpha)*np.sin(beta))
    Z.append(z - 0.5*mutation_scale*length*np.cos(alpha))

    X.append(x - 0.5*mutation_scale*length*np.sin(alpha)*np.cos(beta)
             + 0.1*length*np.cos(alpha)*np.cos(beta))
    Y.append(y - 0.5*mutation_scale*length*np.sin(alpha)*np.sin(beta)
             + 0.1*length*np.cos(alpha)*np.sin(beta))
    Z.append(z - 0.5*mutation_scale*length*np.cos(alpha) - 0.1*length*np.sin(alpha))

    X.append(x - 0.5*mutation_scale*length*np.sin(alpha)*np.cos(beta)
             - 0.1*length*np.cos(alpha)*np.cos(beta))
    Y.append(y - 0.5*mutation_scale*length*np.sin(alpha)*np.sin(beta)
             - 0.1*length*np.cos(alpha)*np.sin(beta))
    Z.append(z - 0.5*mutation_scale*length*np.cos(alpha) - 0.1*length*np.sin(alpha))

    return X,Y,Z


def extreme_distribution_surface():
    country = 'COAST'
    data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(country))
    start_date = pd.to_datetime('2021/01/01/00')
    end_date = pd.to_datetime('2024/12/31/23')
    data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

    load = np.array(data['Load'])/25000
    temperature = np.array(data['Temperature'])/40

    T_i_list = np.array([np.mean(temperature[24 * i:24 * (i + 1)])
                         for i in range(temperature.shape[0] // 24)])
    load_i_list = np.array([np.max(load[24 * i:24 * (i + 1)])
                            for i in range(temperature.shape[0] // 24)])

    # 使用核密度估计计算概率密度
    xy = np.vstack([T_i_list, load_i_list])
    kde = gaussian_kde(xy)

    # 创建网格以评估密度
    x_grid = np.linspace(0, 40, 100)/40
    y_grid = np.linspace(5000, 30000, 100)/25000
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    z_mesh = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)

    # 创建一个 3D 图形
    fig = plt.figure(figsize=(6, 5))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    # 绘制透明曲面图
    ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', alpha=0, edgecolor='none')

    # 定义渐变颜色映射
    #cmap = cm.get_cmap("plasma")  # 从紫色到黄色的渐变
    cmap = pypalettes.load_cmap("Bmsurface")
    #cmap = pypalettes.load_cmap("BuRd")
    #cmap = pypalettes.load_cmap("wailord")

    # 绘制等势线（每个温度对应一条线）
    for i, temp in enumerate(np.linspace(-5, 45, 50)/40):
        z_value = kde(np.vstack([temp * np.ones_like(y_grid), y_grid.ravel()])).reshape(y_grid.shape)
        color = cmap(abs((50/2-i) / 25))  # 使用渐变颜色
        ax.plot(np.full_like(y_grid, temp), y_grid, z_value, color=color, linewidth=1.5)
    # 设置标签
    ax.set_xlabel('Normalized Temperature', fontsize=12, labelpad=13)
    ax.set_ylabel('Normalized Load', fontsize=12, labelpad=13)
    ax.set_zlabel('Probability Density', fontsize=12, labelpad=5)
    #ax.set_zticks([])
    #ax.set_xticklabels(ax.get_xticks(), fontsize=11)
    #ax.set_yticklabels(ax.get_yticks(), fontsize=11)
    #ax.set_zticklabels(ax.get_zticks(), fontsize=11)

    # plot arrow
    #ax.quiver(0, 25000, 1e-5, 0, 0, 0.5e-5, color='b', arrow_length_ratio=0.5)

    start_x, start_y, start_z = 0.12, 0.7, 10  # 起点设置为 z=14
    u, v, w = 0, 0, -8  # 方向向量
    plotArrow3D(ax, start_x, start_y, start_z, u, v, w, arrowLen=0.5, mutation_scale=10, color='darkcyan')
    ax.text(start_x+0.2, start_y, start_z + 2, "Coldwaves", color='darkcyan', fontsize=12, fontweight='bold')

    start_x, start_y, start_z = 1, 0.5, 10  # 起点设置为 z=14
    u, v, w = 0.02, 0.56, -8  # 方向向量
    plotArrow3D(ax, start_x, start_y, start_z, u, v, w, arrowLen=0.5, mutation_scale=10, color='crimson')
    ax.text(start_x + 0.2, start_y, start_z + 2, "Heatwaves", color='crimson', fontsize=12, fontweight='bold')

    # 去掉背景网格
    ax.grid(False)

    norm = Normalize(vmin=-5, vmax=45)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0, shrink=0.5, aspect=10)
    cbar.ax.text(0.5, 1.05, 'Temperature\nDeviation', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=12)

    # 调整视角
    ax.view_init(elev=20, azim=55)
    #ax.view_init(elev=20, azim=75)

    # 显示图形
    #plt.title('3D Surface with Temperature Contours')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

#extreme_distribution_surface()


def save_extreme_ramping_rate():

    texas_common_rr = []
    texas_heatwave_rr = []
    texas_coldwave_rr = []
    for country in ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C', 'SOUTH_C',
                    'SOUTHERN', 'WEST']:
        #country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2021/01/01/00')
        end_date = pd.to_datetime('2024/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])


        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 23):
                rr += abs(load[24 * (i + 30) + j] - load[24 * (i + 30) + j + 1])
            if ECF[i] == 0 and EHF[i] == 0:
                texas_common_rr.append(rr)

            elif EHF[i] != 0:
                texas_heatwave_rr.append(rr)

            else:
                texas_coldwave_rr.append(rr)
    # Europe
    EU_common_rr = []
    EU_heatwave_rr = []
    EU_coldwave_rr = []
    for country in ['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland']:
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

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 23):
                rr += abs(load[24 * (i + 30) + j] - load[24 * (i + 30) + j + 1])
            if ECF[i] == 0 and EHF[i] == 0:
                EU_common_rr.append(rr)

            elif EHF[i] != 0:
                EU_heatwave_rr.append(rr)

            else:
                EU_coldwave_rr.append(rr)

    ############## Guangdong
    GD_common_rr = []
    GD_heatwave_rr = []
    GD_coldwave_rr = []
    for country in ['chaozhou', 'dongguan', 'foshan', 'guangzhou', 'heyuan', 'huizhou',
                   'jiangmen', 'jieyang', 'maoming', 'meizhou', 'qingyuan', 'shantou',
                   'shanwei', 'shaoguan', 'shenzhen', 'yangjiang', 'yunfu', 'zhanjiang',
                   'zhaoqing', 'zhongshan', 'zhuhai']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/GuangDong_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2020/01/01/00')
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[
            (pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 23):
                rr += abs(load[24 * (i + 30) + j] - load[24 * (i + 30) + j + 1])
            if ECF[i] == 0 and EHF[i] == 0:
                GD_common_rr.append(rr)

            elif EHF[i] != 0:
                GD_heatwave_rr.append(rr)

            else:
                GD_coldwave_rr.append(rr)





    ################ India
    ID_common_rr = []
    ID_heatwave_rr = []
    ID_coldwave_rr = []
    for country in ['Maharashtra_data_2017_2023',
                    'Delhi_data_2017_2023']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/India_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2019/01/01/00')
        end_date = pd.to_datetime('2021/12/31/23')
        data = data[
            (pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 23):
                rr += abs(load[24 * (i + 30) + j] - load[24 * (i + 30) + j + 1])
            if ECF[i] == 0 and EHF[i] == 0:
                ID_common_rr.append(rr)

            elif EHF[i] != 0:
                ID_heatwave_rr.append(rr)

            else:
                ID_coldwave_rr.append(rr)

    ################ PJM
    PJM_common_rr = []
    PJM_heatwave_rr = []
    PJM_coldwave_rr = []

    for country in ['Allegheny Power System',
                'American Electric Power Co., Inc', 'Atlantic Electric Company',
                'Baltimore Gas and Electric Company',
                'ComEd', 'Dayton Power and Light Company', 'Delmarva Power and Light',
                'Dominion Energy', 'Duke Energy Ohio', 'Duquesne Light',
                'East Kentucky Power Coop', 'First Energy - Pennsylvania Electric Company',
                'Jersey Central Power and Light Company', 'Metropolitan Edison Company',
                'Orion Energy', 'Pennsylvania Electric Company',
                'Pennsylvania Power and Light Company',
                'Potomac Electric Power',
                'Public Service Electric and Gas Company']:
        # country = 'COAST'
        data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(country), header=0,
                           usecols=['Date_Hour', 'Load', 'Temperature'])

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

        start_date = pd.to_datetime('2021/03/28/00')  ## Thursday
        end_date = pd.to_datetime('2023/05/31/23')
        data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]

        data['Date_Hour'] = pd.to_datetime(data['Date_Hour'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 23):
                rr += abs(load[24 * (i + 30) + j] - load[24 * (i + 30) + j + 1])
            if ECF[i] == 0 and EHF[i] == 0:
                PJM_common_rr.append(rr)

            elif EHF[i] != 0:
                PJM_heatwave_rr.append(rr)

            else:
                PJM_coldwave_rr.append(rr)

    ################ Hunan
    HN_common_rr = []
    HN_heatwave_rr = []
    HN_coldwave_rr = []

    for country in ['娄底', '岳阳',
                    '常德',
                    '张家界',
                    '怀化',
                    '株洲',
                    '永州',
                    '湘潭',
                    '湘西',
                    '益阳',
                    '衡阳',
                    '邵阳',
                    '郴州',
                    '长沙']:
        # country = 'COAST'
        data = pd.read_csv('../Data/reformed_data_updated/hunan_data_reformed/{}.csv'.format(country), header=0, usecols=['Date', 'load', 'temp'])


        start_date = pd.to_datetime('2021/01/01/00')  ## Thursday
        end_date = pd.to_datetime('2023/09/30/23')
        data = data[(pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)]

        data['Date'] = pd.to_datetime(data['Date'])  # 确保 Data_Hour 列为 datetime 类型

        load = np.array(data['load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['temp'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 23):
                rr += abs(load[24 * (i + 30) + j] - load[24 * (i + 30) + j + 1])
            if ECF[i] == 0 and EHF[i] == 0:
                HN_common_rr.append(rr)

            elif EHF[i] != 0:
                HN_heatwave_rr.append(rr)

            else:
                HN_coldwave_rr.append(rr)


    data = {
        'Group': ['EU']*len(EU_common_rr)+['GD']*len(GD_common_rr)+
                 ['India']*len(ID_common_rr)+['Texas']*len(texas_common_rr)+
                 ['PJM']*len(PJM_common_rr)+['Hunan']*len(HN_common_rr),   # 三个组
        'Value': EU_common_rr+GD_common_rr+ID_common_rr+texas_common_rr+PJM_common_rr+
                 HN_common_rr,
        'HUE': ['EU']*len(EU_common_rr)+['GD']*len(GD_common_rr)+
                 ['India']*len(ID_common_rr)+['Texas']*len(texas_common_rr)+
                 ['PJM']*len(PJM_common_rr)+['Hunan']*len(HN_common_rr)}
    df = pd.DataFrame(data)
    df.to_excel('ramping_common.xlsx', index=False)

    data = {
        'Group': ['EU'] * len(EU_coldwave_rr) + ['GD'] * len(GD_heatwave_rr) +
                 ['India'] * len(ID_heatwave_rr) + ['Texas'] * len(texas_heatwave_rr) +
                 ['PJM'] * len(PJM_heatwave_rr) + ['Hunan'] * len(HN_coldwave_rr),  # 三个组
        'Value': EU_coldwave_rr + GD_heatwave_rr + ID_heatwave_rr + texas_heatwave_rr + PJM_heatwave_rr +
                 HN_coldwave_rr,
        'HUE': ['EU'] * len(EU_coldwave_rr) + ['GD'] * len(GD_heatwave_rr) +
               ['India'] * len(ID_heatwave_rr) + ['Texas'] * len(texas_heatwave_rr) +
               ['PJM'] * len(PJM_heatwave_rr) + ['Hunan'] * len(HN_coldwave_rr)}

    df = pd.DataFrame(data)
    df.to_excel('ramping_extreme.xlsx', index=False)


def save_extreme_values():

    texas_common_rr = []
    texas_heatwave_rr = []
    texas_coldwave_rr = []
    for country in ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C', 'SOUTH_C',
                    'SOUTHERN', 'WEST']:
        #country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2021/01/01/00')
        end_date = pd.to_datetime('2024/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])


        for i in range(EHF.shape[0]):
            rr = max(load[24 * (i + 30):24 * (i + 31)])
            if ECF[i] == 0 and EHF[i] == 0:
                texas_common_rr.append(rr)

            elif EHF[i] != 0:
                texas_heatwave_rr.append(rr)

            else:
                texas_coldwave_rr.append(rr)
    # Europe
    EU_common_rr = []
    EU_heatwave_rr = []
    EU_coldwave_rr = []
    for country in ['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland']:
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

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = max(load[24 * (i + 30):24 * (i + 31)])
            if ECF[i] == 0 and EHF[i] == 0:
                EU_common_rr.append(rr)

            elif EHF[i] != 0:
                EU_heatwave_rr.append(rr)

            else:
                EU_coldwave_rr.append(rr)

    ############## Guangdong
    GD_common_rr = []
    GD_heatwave_rr = []
    GD_coldwave_rr = []
    for country in ['chaozhou', 'dongguan', 'foshan', 'guangzhou', 'heyuan', 'huizhou',
                   'jiangmen', 'jieyang', 'maoming', 'meizhou', 'qingyuan', 'shantou',
                   'shanwei', 'shaoguan', 'shenzhen', 'yangjiang', 'yunfu', 'zhanjiang',
                   'zhaoqing', 'zhongshan', 'zhuhai']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/GuangDong_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2020/01/01/00')
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[
            (pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = max(load[24 * (i + 30):24 * (i + 31)])
            if ECF[i] == 0 and EHF[i] == 0:
                GD_common_rr.append(rr)

            elif EHF[i] != 0:
                GD_heatwave_rr.append(rr)

            else:
                GD_coldwave_rr.append(rr)





    ################ India
    ID_common_rr = []
    ID_heatwave_rr = []
    ID_coldwave_rr = []
    for country in ['Maharashtra_data_2017_2023',
                    'Delhi_data_2017_2023']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/India_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2019/01/01/00')
        end_date = pd.to_datetime('2021/12/31/23')
        data = data[
            (pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = max(load[24 * (i + 30):24 * (i + 31)])
            if ECF[i] == 0 and EHF[i] == 0:
                ID_common_rr.append(rr)

            elif EHF[i] != 0:
                ID_heatwave_rr.append(rr)

            else:
                ID_coldwave_rr.append(rr)

    ################ PJM
    PJM_common_rr = []
    PJM_heatwave_rr = []
    PJM_coldwave_rr = []

    for country in ['Allegheny Power System',
                'American Electric Power Co., Inc', 'Atlantic Electric Company',
                'Baltimore Gas and Electric Company',
                'ComEd', 'Dayton Power and Light Company', 'Delmarva Power and Light',
                'Dominion Energy', 'Duke Energy Ohio', 'Duquesne Light',
                'East Kentucky Power Coop', 'First Energy - Pennsylvania Electric Company',
                'Jersey Central Power and Light Company', 'Metropolitan Edison Company',
                'Orion Energy', 'Pennsylvania Electric Company',
                'Pennsylvania Power and Light Company',
                'Potomac Electric Power',
                'Public Service Electric and Gas Company']:
        # country = 'COAST'
        data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(country), header=0,
                           usecols=['Date_Hour', 'Load', 'Temperature'])

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

        start_date = pd.to_datetime('2022/01/01/00')  ## Thursday
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]

        data['Date_Hour'] = pd.to_datetime(data['Date_Hour'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = max(load[24 * (i + 30):24 * (i + 31)])
            if ECF[i] == 0 and EHF[i] == 0:
                PJM_common_rr.append(rr)

            elif EHF[i] != 0:
                PJM_heatwave_rr.append(rr)

            else:
                PJM_coldwave_rr.append(rr)

    ################ Hunan
    HN_common_rr = []
    HN_heatwave_rr = []
    HN_coldwave_rr = []

    for country in ['娄底', '岳阳',
                    '常德',
                    '张家界',
                    '怀化',
                    '株洲',
                    '永州',
                    '湘潭',
                    '湘西',
                    '益阳',
                    '衡阳',
                    '邵阳',
                    '郴州',
                    '长沙']:
        # country = 'COAST'
        data = pd.read_csv('../Data/reformed_data_updated/hunan_data_reformed/{}.csv'.format(country), header=0, usecols=['Date', 'load', 'temp'])


        start_date = pd.to_datetime('2021/01/01/00')  ## Thursday
        end_date = pd.to_datetime('2023/09/30/23')
        data = data[(pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)]

        data['Date'] = pd.to_datetime(data['Date'])  # 确保 Data_Hour 列为 datetime 类型

        load = np.array(data['load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['temp'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = max(load[24 * (i + 30):24 * (i + 31)])
            if ECF[i] == 0 and EHF[i] == 0:
                HN_common_rr.append(rr)

            elif EHF[i] != 0:
                HN_heatwave_rr.append(rr)

            else:
                HN_coldwave_rr.append(rr)


    data = {
        'Group': ['EU']*len(EU_common_rr)+['GD']*len(GD_common_rr)+
                 ['India']*len(ID_common_rr)+['Texas']*len(texas_common_rr)+
                 ['PJM']*len(PJM_common_rr)+['Hunan']*len(HN_common_rr),   # 三个组
        'Value': EU_common_rr+GD_common_rr+ID_common_rr+texas_common_rr+PJM_common_rr+
                 HN_common_rr,
        'HUE': ['EU']*len(EU_common_rr)+['GD']*len(GD_common_rr)+
                 ['India']*len(ID_common_rr)+['Texas']*len(texas_common_rr)+
                 ['PJM']*len(PJM_common_rr)+['Hunan']*len(HN_common_rr)}
    df = pd.DataFrame(data)
    df.to_excel('extreme_values_common.xlsx', index=False)

    data = {
        'Group': ['EU'] * len(EU_coldwave_rr) + ['GD'] * len(GD_heatwave_rr) +
                 ['India'] * len(ID_heatwave_rr) + ['Texas'] * len(texas_heatwave_rr) +
                 ['PJM'] * len(PJM_heatwave_rr) + ['Hunan'] * len(HN_coldwave_rr),  # 三个组
        'Value': EU_coldwave_rr + GD_heatwave_rr + ID_heatwave_rr + texas_heatwave_rr + PJM_heatwave_rr +
                 HN_coldwave_rr,
        'HUE': ['EU'] * len(EU_coldwave_rr) + ['GD'] * len(GD_heatwave_rr) +
               ['India'] * len(ID_heatwave_rr) + ['Texas'] * len(texas_heatwave_rr) +
               ['PJM'] * len(PJM_heatwave_rr) + ['Hunan'] * len(HN_coldwave_rr)}

    df = pd.DataFrame(data)
    df.to_excel('extreme_values_extreme.xlsx', index=False)


def save_extreme_energy():

    texas_common_rr = []
    texas_heatwave_rr = []
    texas_coldwave_rr = []
    for country in ['COAST', 'EAST', 'FAR_WEST', 'NORTH', 'NORTH_C', 'SOUTH_C',
                    'SOUTHERN', 'WEST']:
        #country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2021/01/01/00')
        end_date = pd.to_datetime('2024/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])


        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 24):
                rr += load[24 * (i + 30) + j]
            if ECF[i] == 0 and EHF[i] == 0:
                texas_common_rr.append(rr)

            elif EHF[i] != 0:
                texas_heatwave_rr.append(rr)

            else:
                texas_coldwave_rr.append(rr)




    ############# Europe
    EU_common_rr = []
    EU_heatwave_rr = []
    EU_coldwave_rr = []
    for country in ['Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
                     'Germany', 'Hungary', 'Ireland', 'Italy',
                      'Lithuania', 'Latvia', 'Netherlands', 'Norway',
                      'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland']:
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

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 24):
                rr += load[24 * (i + 30) + j]
            if ECF[i] == 0 and EHF[i] == 0:
                EU_common_rr.append(rr)

            elif EHF[i] != 0:
                EU_heatwave_rr.append(rr)

            else:
                EU_coldwave_rr.append(rr)

    ############## Guangdong
    GD_common_rr = []
    GD_heatwave_rr = []
    GD_coldwave_rr = []
    for country in ['chaozhou', 'dongguan', 'foshan', 'guangzhou', 'heyuan', 'huizhou',
                   'jiangmen', 'jieyang', 'maoming', 'meizhou', 'qingyuan', 'shantou',
                   'shanwei', 'shaoguan', 'shenzhen', 'yangjiang', 'yunfu', 'zhanjiang',
                   'zhaoqing', 'zhongshan', 'zhuhai']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/GuangDong_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2020/01/01/00')
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[
            (pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 24):
                rr += load[24 * (i + 30) + j]
            if ECF[i] == 0 and EHF[i] == 0:
                GD_common_rr.append(rr)

            elif EHF[i] != 0:
                GD_heatwave_rr.append(rr)

            else:
                GD_coldwave_rr.append(rr)





    ################ India
    ID_common_rr = []
    ID_heatwave_rr = []
    ID_coldwave_rr = []
    for country in ['Maharashtra_data_2017_2023',
                    'Delhi_data_2017_2023']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/India_data_reformed/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2019/01/01/00')
        end_date = pd.to_datetime('2021/12/31/23')
        data = data[
            (pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 24):
                rr += load[24 * (i + 30) + j]
            if ECF[i] == 0 and EHF[i] == 0:
                ID_common_rr.append(rr)

            elif EHF[i] != 0:
                ID_heatwave_rr.append(rr)

            else:
                ID_coldwave_rr.append(rr)

    ################ PJM
    PJM_common_rr = []
    PJM_heatwave_rr = []
    PJM_coldwave_rr = []

    for country in ['Allegheny Power System',
                'American Electric Power Co., Inc', 'Atlantic Electric Company',
                'Baltimore Gas and Electric Company',
                'ComEd', 'Dayton Power and Light Company', 'Delmarva Power and Light',
                'Dominion Energy', 'Duke Energy Ohio', 'Duquesne Light',
                'East Kentucky Power Coop', 'First Energy - Pennsylvania Electric Company',
                'Jersey Central Power and Light Company', 'Metropolitan Edison Company',
                'Orion Energy', 'Pennsylvania Electric Company',
                'Pennsylvania Power and Light Company',
                'Potomac Electric Power',
                'Public Service Electric and Gas Company']:
        # country = 'COAST'
        data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(country), header=0,
                           usecols=['Date_Hour', 'Load', 'Temperature'])

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

        start_date = pd.to_datetime('2022/01/01/00')  ## Thursday
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]

        data['Date_Hour'] = pd.to_datetime(data['Date_Hour'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 24):
                rr += load[24 * (i + 30) + j]
            if ECF[i] == 0 and EHF[i] == 0:
                PJM_common_rr.append(rr)

            elif EHF[i] != 0:
                PJM_heatwave_rr.append(rr)

            else:
                PJM_coldwave_rr.append(rr)

    ################ Hunan
    HN_common_rr = []
    HN_heatwave_rr = []
    HN_coldwave_rr = []

    for country in ['娄底', '岳阳',
                    '常德',
                    '张家界',
                    '怀化',
                    '株洲',
                    '永州',
                    '湘潭',
                    '湘西',
                    '益阳',
                    '衡阳',
                    '邵阳',
                    '郴州',
                    '长沙']:
        # country = 'COAST'
        data = pd.read_csv('../Data/reformed_data_updated/hunan_data_reformed/{}.csv'.format(country), header=0, usecols=['Date', 'load', 'temp'])


        start_date = pd.to_datetime('2021/01/01/00')  ## Thursday
        end_date = pd.to_datetime('2023/09/30/23')
        data = data[(pd.to_datetime(data['Date']) >= start_date) & (pd.to_datetime(data['Date']) <= end_date)]

        data['Date'] = pd.to_datetime(data['Date'])  # 确保 Data_Hour 列为 datetime 类型

        load = np.array(data['load'])
        load = (load - min(load)) / (max(load) - min(load))
        temperature = np.array(data['temp'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature))

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        # define the coldwave index
        T_05 = np.percentile(T_i_list, 5)
        ECI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_05
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        ECI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        ECF = np.array([min(0, -ECI_sig[i] * min(-1, ECI_accl[i])) for i in range(ECI_sig.shape[0])])

        # define the heatwave index
        T_95 = np.percentile(T_i_list, 95)
        EHI_sig = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - T_95
                            for i in range(T_i_list.shape[0] - 3 - 30)])
        EHI_accl = np.array([np.mean(T_i_list[i + 30:i + 30 + 3]) - np.mean(T_i_list[i:i + 30])
                             for i in range(T_i_list.shape[0] - 3 - 30)])
        EHF = np.array([max(0, EHI_sig[i] * max(1, EHI_accl[i])) for i in range(EHI_sig.shape[0])])

        for i in range(EHF.shape[0]):
            rr = 0
            for j in range(0, 24):
                rr += load[24 * (i + 30) + j]
            if ECF[i] == 0 and EHF[i] == 0:
                HN_common_rr.append(rr)

            elif EHF[i] != 0:
                HN_heatwave_rr.append(rr)

            else:
                HN_coldwave_rr.append(rr)


    data = {
        'Group': ['EU']*len(EU_common_rr)+['GD']*len(GD_common_rr)+
                 ['India']*len(ID_common_rr)+['Texas']*len(texas_common_rr)+
                 ['PJM']*len(PJM_common_rr)+['Hunan']*len(HN_common_rr),   # 三个组
        'Value': EU_common_rr+GD_common_rr+ID_common_rr+texas_common_rr+PJM_common_rr+
                 HN_common_rr,
        'HUE': ['EU']*len(EU_common_rr)+['GD']*len(GD_common_rr)+
                 ['India']*len(ID_common_rr)+['Texas']*len(texas_common_rr)+
                 ['PJM']*len(PJM_common_rr)+['Hunan']*len(HN_common_rr)}
    df = pd.DataFrame(data)
    df.to_excel('energy_common.xlsx', index=False)

    data = {
        'Group': ['EU'] * len(EU_coldwave_rr) + ['GD'] * len(GD_heatwave_rr) +
                 ['India'] * len(ID_heatwave_rr) + ['Texas'] * len(texas_heatwave_rr) +
                 ['PJM'] * len(PJM_heatwave_rr) + ['Hunan'] * len(HN_coldwave_rr),  # 三个组
        'Value': EU_coldwave_rr + GD_heatwave_rr + ID_heatwave_rr + texas_heatwave_rr + PJM_heatwave_rr +
                 HN_coldwave_rr,
        'HUE': ['EU'] * len(EU_coldwave_rr) + ['GD'] * len(GD_heatwave_rr) +
               ['India'] * len(ID_heatwave_rr) + ['Texas'] * len(texas_heatwave_rr) +
               ['PJM'] * len(PJM_heatwave_rr) + ['Hunan'] * len(HN_coldwave_rr)}

    df = pd.DataFrame(data)
    df.to_excel('energy_extreme.xlsx', index=False)


#save_extreme_ramping_rate()
#save_extreme_values()
#save_extreme_energy()


def plot_extreme_boxes():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'



    fig, ax = plt.subplots(3, 1, figsize=(6, 6))

    ### ramping_common
    df = pd.read_excel('ramping_common.xlsx')


    custom_palette = {'GD': '#A8CDECFF', 'PJM': '#F6955EFF', 'Texas': '#682C37FF',
                      'India': '#9B6981FF', 'Hunan': '#7887A4FF', 'EU': '#A89F8EFF'}
    sns.violinplot(x='Group', y='Value', hue='HUE', data=df, ax=ax[0], inner='quartile', palette=custom_palette,
                   linewidth=1.5)

    ### ramping_extreme
    df = pd.read_excel('ramping_extreme.xlsx')
    sns.violinplot(x='Group', y='Value', hue='HUE', data=df, ax=ax[0], inner='quartile', palette=custom_palette,
                   linewidth=1.5, alpha=0.3, linestyle='--')

    ### extreme_values_common
    df = pd.read_excel('extreme_values_common.xlsx')
    sns.violinplot(x='Group', y='Value', hue='HUE', data=df, ax=ax[1], inner='quartile', palette=custom_palette,
                   linewidth=1.5)


    ### extreme_values_extreme
    df = pd.read_excel('extreme_values_extreme.xlsx')
    sns.violinplot(x='Group', y='Value', hue='HUE', data=df, ax=ax[1], inner='quartile', palette=custom_palette,
                   linewidth=1.5, alpha=0.3)

    ### energy_common
    df = pd.read_excel('energy_common.xlsx')
    sns.violinplot(x='Group', y='Value', hue='HUE', data=df, ax=ax[2], inner=None, palette=custom_palette,
                   linewidth=1.5)

    sns.pointplot(
        x='Group', y='Value', hue='HUE',
        data=df, ax=ax[2],
        palette=custom_palette,
        estimator=np.mean,  # 显示均值
        markers="o",  # 点样式
        scale=1.2,  # 点大小
        ci=None,  # 不显示置信区间
        linestyles="",  # 不显示连线
    )

    ### extreme_values_extreme
    df = pd.read_excel('energy_extreme.xlsx')
    sns.violinplot(x='Group', y='Value', hue='HUE', data=df, ax=ax[2], inner='box', palette=custom_palette,
                   linewidth=1.5, alpha=0.3)

    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    ax[0].set_ylabel('Ramping Rate')
    ax[1].set_ylabel('Maximum Value')
    ax[2].set_ylabel('Energy')
    ax[2].set_xlabel('Datasets')

    plt.show()




def plot_extreme_boxes_2():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots(3, 1, figsize=(6, 6))
    custom_palette = ['#A8CDECFF', '#F6955EFF', 'peru',
                      '#9B6981FF', '#7887A4FF', '#A89F8EFF']  # 示例颜色，替换成你的调色板

    ### ramping_common
    df = pd.read_excel('ramping_common.xlsx')

    # 按 Group 和 HUE 分组数据
    groups = df['Group'].unique()
    hues = df['HUE'].unique()
    print("Group 类别:", df['Group'].unique())
    print("HUE 类别:", df['HUE'].unique())


    # 存储 violin 数据
    violin_data = []
    positions = []
    colors = []

    pos = 1
    for i, group in enumerate(groups):
        for j, hue in enumerate(hues):
            subset = df[(df['Group'] == group) & (df['HUE'] == hue)]
            values = subset['Value'].dropna().values
            if len(values) > 0:  # 只添加非空数据
                violin_data.append(values)
                positions.append(pos)
                colors.append(custom_palette[j])
                pos += 1
        pos += 1  # 增加间距

    #print(violin_data)

    # 绘制 violinplot
    violins = ax[0].violinplot(
        violin_data,
        positions=positions,
        bw_method="silverman",
        showextrema=False,
        showmeans=True,  # 显示均值（代替 inner='quartile'）
        showmedians=False,  # 不显示中位数
        quantiles=None,  # 不显示分位数
        widths=0.8,  # 调整宽度
    )

    violins['cmeans'].set_color('black')  # 均值线颜色
    violins['cmeans'].set_linewidth(1)  # 线宽
    violins['cmeans'].set_linestyle('-')

    # 设置颜色和样式
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)  # 透明度
        pc.set_linestyle('-')  # 线型
        pc.set_linewidth(0.8)  # 线宽


    ######### ramping_extreme
    df = pd.read_excel('ramping_extreme.xlsx')

    # 按 Group 和 HUE 分组数据
    groups = df['Group'].unique()
    hues = df['HUE'].unique()
    print("Group 类别:", df['Group'].unique())
    print("HUE 类别:", df['HUE'].unique())

    # 存储 violin 数据
    violin_data = []
    positions = []
    colors = []

    pos = 1
    for i, group in enumerate(groups):
        for j, hue in enumerate(hues):
            subset = df[(df['Group'] == group) & (df['HUE'] == hue)]
            values = subset['Value'].dropna().values
            if len(values) > 0:  # 只添加非空数据
                violin_data.append(values)
                positions.append(pos)
                colors.append(custom_palette[j])
                pos += 1
        pos += 1  # 增加间距

    # print(violin_data)

    # 绘制 violinplot
    violins = ax[0].violinplot(
        violin_data,
        positions=positions,
        bw_method="silverman",
        showextrema=False,
        showmeans=True,  # 显示均值（代替 inner='quartile'）
        showmedians=False,  # 不显示中位数
        quantiles=None,  # 不显示分位数
        widths=0.8,  # 调整宽度
    )

    violins['cmeans'].set_color('black')  # 均值线颜色
    violins['cmeans'].set_linewidth(1)  # 线宽
    violins['cmeans'].set_linestyle('--')

    # 设置颜色和样式
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)  # 透明度
        pc.set_linestyle('--')  # 线型
        pc.set_linewidth(0.8)  # 线宽





    ### extreme_values_common
    df = pd.read_excel('extreme_values_common.xlsx')

    # 按 Group 和 HUE 分组数据
    groups = df['Group'].unique()
    hues = df['HUE'].unique()
    print("Group 类别:", df['Group'].unique())
    print("HUE 类别:", df['HUE'].unique())

    # 存储 violin 数据
    violin_data = []
    positions = []
    colors = []

    pos = 1
    for i, group in enumerate(groups):
        for j, hue in enumerate(hues):
            subset = df[(df['Group'] == group) & (df['HUE'] == hue)]
            values = subset['Value'].dropna().values
            if len(values) > 0:  # 只添加非空数据
                violin_data.append(values)
                positions.append(pos)
                colors.append(custom_palette[j])
                pos += 1
        pos += 1  # 增加间距

    # print(violin_data)

    # 绘制 violinplot
    violins = ax[1].violinplot(
        violin_data,
        positions=positions,
        bw_method="silverman",
        showextrema=False,
        showmeans=True,  # 显示均值（代替 inner='quartile'）
        showmedians=False,  # 不显示中位数
        quantiles=None,  # 不显示分位数
        widths=0.8,  # 调整宽度
    )

    violins['cmeans'].set_color('black')  # 均值线颜色
    violins['cmeans'].set_linewidth(1)  # 线宽
    violins['cmeans'].set_linestyle('-')

    # 设置颜色和样式
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)  # 透明度
        pc.set_linestyle('-')  # 线型
        pc.set_linewidth(0.8)  # 线宽

    ######### extreme_values_extreme
    df = pd.read_excel('extreme_values_extreme.xlsx')

    # 按 Group 和 HUE 分组数据
    groups = df['Group'].unique()
    hues = df['HUE'].unique()
    print("Group 类别:", df['Group'].unique())
    print("HUE 类别:", df['HUE'].unique())

    # 存储 violin 数据
    violin_data = []
    positions = []
    colors = []

    pos = 1
    for i, group in enumerate(groups):
        for j, hue in enumerate(hues):
            subset = df[(df['Group'] == group) & (df['HUE'] == hue)]
            values = subset['Value'].dropna().values
            if len(values) > 0:  # 只添加非空数据
                violin_data.append(values)
                positions.append(pos)
                colors.append(custom_palette[j])
                pos += 1
        pos += 1  # 增加间距

    # print(violin_data)

    # 绘制 violinplot
    violins = ax[1].violinplot(
        violin_data,
        positions=positions,
        bw_method="silverman",
        showextrema=False,
        showmeans=True,  # 显示均值（代替 inner='quartile'）
        showmedians=False,  # 不显示中位数
        quantiles=None,  # 不显示分位数
        widths=0.8,  # 调整宽度
    )

    violins['cmeans'].set_color('black')  # 均值线颜色
    violins['cmeans'].set_linewidth(1)  # 线宽
    violins['cmeans'].set_linestyle('--')

    # 设置颜色和样式
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)  # 透明度
        pc.set_linestyle('--')  # 线型
        pc.set_linewidth(0.8)  # 线宽




    ### energy_common
    df = pd.read_excel('energy_common.xlsx')

    # 按 Group 和 HUE 分组数据
    groups = df['Group'].unique()
    hues = df['HUE'].unique()
    print("Group 类别:", df['Group'].unique())
    print("HUE 类别:", df['HUE'].unique())

    # 存储 violin 数据
    violin_data = []
    positions = []
    colors = []

    pos = 1
    for i, group in enumerate(groups):
        for j, hue in enumerate(hues):
            subset = df[(df['Group'] == group) & (df['HUE'] == hue)]
            values = subset['Value'].dropna().values
            if len(values) > 0:  # 只添加非空数据
                violin_data.append(values)
                positions.append(pos)
                colors.append(custom_palette[j])
                pos += 1
        pos += 1  # 增加间距

    # print(violin_data)

    # 绘制 violinplot
    violins = ax[2].violinplot(
        violin_data,
        positions=positions,
        bw_method="silverman",
        showextrema=False,
        showmeans=True,  # 显示均值（代替 inner='quartile'）
        showmedians=False,  # 不显示中位数
        quantiles=None,  # 不显示分位数
        widths=0.8,  # 调整宽度
    )

    violins['cmeans'].set_color('black')  # 均值线颜色
    violins['cmeans'].set_linewidth(1)  # 线宽
    violins['cmeans'].set_linestyle('-')

    # 设置颜色和样式
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(1)  # 透明度
        pc.set_linestyle('-')  # 线型
        pc.set_linewidth(0.8)  # 线宽

    ######### energy_extreme
    df = pd.read_excel('energy_extreme.xlsx')

    # 按 Group 和 HUE 分组数据
    groups = df['Group'].unique()
    hues = df['HUE'].unique()
    print("Group 类别:", df['Group'].unique())
    print("HUE 类别:", df['HUE'].unique())

    # 存储 violin 数据
    violin_data = []
    positions = []
    colors = []

    pos = 1
    for i, group in enumerate(groups):
        for j, hue in enumerate(hues):
            subset = df[(df['Group'] == group) & (df['HUE'] == hue)]
            values = subset['Value'].dropna().values
            if len(values) > 0:  # 只添加非空数据
                violin_data.append(values)
                positions.append(pos)
                colors.append(custom_palette[j])
                pos += 1
        pos += 1  # 增加间距

    # print(violin_data)

    # 绘制 violinplot
    violins = ax[2].violinplot(
        violin_data,
        positions=positions,
        bw_method="silverman",
        showextrema=False,
        showmeans=True,  # 显示均值（代替 inner='quartile'）
        showmedians=False,  # 不显示中位数
        quantiles=None,  # 不显示分位数
        widths=0.8,  # 调整宽度
    )

    violins['cmeans'].set_color('black')  # 均值线颜色
    violins['cmeans'].set_linewidth(1)  # 线宽
    violins['cmeans'].set_linestyle('--')

    # 设置颜色和样式
    for i, pc in enumerate(violins['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)  # 透明度
        pc.set_linestyle('--')  # 线型
        pc.set_linewidth(0.8)  # 线宽




    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    ax[0].set_ylabel('Ramping Rate', labelpad=10)
    ax[1].set_ylabel('Maximum Value', labelpad=3)
    ax[2].set_ylabel('Energy', labelpad=4)
    ax[2].set_xlabel('Datasets')


    x_labels = ['EU', 'GD', 'India', 'Texas', 'PJM', 'HN']


    for i in range(3):
        ax[i].set_xticks([1, 3, 5, 7, 9, 11])  # 设置刻度位置
        ax[i].tick_params(axis='x', which='major', length=0)
        ax[i].set_xticklabels(x_labels)
        ax[i].minorticks_on()
        ax[i].set_xticks([2, 4, 6, 8, 10, 12], minor=True)
        ax[i].tick_params(axis='x', which='minor', length=5)

    fig.suptitle('Load Patterns Change During Heatwaves and Coldwaves', y=0.94)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#plot_extreme_boxes_2()
#plot_extreme_boxes()

#test_surface()
#extreme_distribution()






def plot_generated_sample_distribution_coldwave(titlesize=12, ticksize=14, labelsize=17,
                                                original_color='#65A2D2',
                                                synthetic_color='#F58E87',
                                                weather_type='coldwave'):
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_coldwave_samples
    # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe
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

        if weather_type == 'coldwave':
            coldwave_samples = generate_coldwave_samples(country, num_samples=400,
                                                         weather_type='coldwave').cpu().detach().numpy()
            coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                        coldwave_samples.shape[1], -1)
        elif weather_type == 'common':
            coldwave_samples = generate_coldwave_samples(country, num_samples=400,
                                                         weather_type='common').cpu().detach().numpy()
            coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                        coldwave_samples.shape[1], -1)

        else:
            coldwave_samples = np.array(torch.rand(1000, 2, 192))


        load_slice_list = np.array(load_slice_list)
        #print(load_slice_list.shape)
        tem_slice_list = np.array(tem_slice_list)

        sample_load_list = np.mean(load_slice_list[:, :-24], axis=1)
        label_load_list = np.mean(load_slice_list[:, -24:], axis=1)
        sample_tem_list = np.mean(tem_slice_list[:, :-24], axis=1)
        generated_sample_load_list = np.mean(coldwave_samples[:, 0, -24:], axis=1)
        generated_sample_tem_list = np.mean(coldwave_samples[:, 1, -24:], axis=1)

        print(sample_load_list.shape)

        df = pd.DataFrame({'Load': sample_load_list, 'Temperature': sample_tem_list,
                           'group': np.repeat('A',sample_load_list.shape[0]) })

        #fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 1行3列，总宽度18，高度6
        #fig.subplots_adjust(wspace=0.4)  # 调整子图之间的水平间距



        #fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        g = sns.jointplot(data=df, x='Load', y='Temperature',
                          color=original_color,
                          height=5.25,  # 控制整个图形的高度
                          ratio=3,  # 主图与边缘图的比例（值越大，主图越小）
                          space=0.3, s=30, lw=0.1, edgecolor=None,
                          marker='o', alpha=0.5,  marginal_ticks=True)

        g.ax_joint.scatter(
            generated_sample_load_list, generated_sample_tem_list,
            color=synthetic_color, marker="o", alpha=1,
            label="Synthetic Samples", s=10, edgecolor=None, lw=0.5
        )

        # 计算统一的坐标范围和分箱
        x_min = 0
        x_max = 0.9
        y_min = 0
        y_max = 0.9
        bins_x = np.linspace(x_min, x_max, 100)
        bins_y = np.linspace(y_min, y_max, 100)
        # 清空原始边缘分布
        g.ax_marg_x.clear()
        g.ax_marg_y.clear()


        # 更新 x 轴边缘分布（包含 new_x）
        #sns.histplot(x=sample_load_list, ax=g.ax_marg_x, bins=bins_x,
        #             color="#682C37FF", alpha=0.5, stat="density")
        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(generated_sample_load_list, bins=bins_x, density=True)
        hist2, _ = np.histogram(sample_load_list, bins=bins_x, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_x.bar(
            x=bins_x[:-1],  # 分箱左边缘作为x坐标
            height=hist1,  # 高度为频数/密度
            width=np.diff(bins_x),  # 宽度为分箱间隔
            color=synthetic_color,  # 原始数据颜色
            alpha=0.5,
            edgecolor=None,
            linewidth=1,
            label="Original"
        )
        g.ax_marg_x.bar(
            x=bins_x[:-1],
            height=hist2,
            width=np.diff(bins_x),
            bottom=hist1,  # 垂直堆叠基准
            color=original_color,  # 生成数据颜色
            alpha=1,
            edgecolor=None,
            linewidth=1,
            label="Generated"
        )


        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(generated_sample_tem_list, bins=bins_y, density=True)
        hist2, _ = np.histogram(sample_tem_list, bins=bins_y, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist1,
            height=np.diff(bin_edges),
            color=synthetic_color,
            alpha=0.5,
            edgecolor=None,  # 边框颜色
            linewidth=1
        )
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist2,
            height=np.diff(bin_edges),
            left=hist1,  # 关键：以hist1为基准堆叠
            color=original_color,
            alpha=1,
            edgecolor=None,  # 边框颜色
            linewidth=1,
        )

        #sns.kdeplot(
        #    y=sample_tem_list,
        #    ax=g.ax_marg_y,
        #    color="#682C37FF",
        #    linewidth=2
        #)

        #sns.kdeplot(
        #    y=generated_sample_tem_list,
        #    ax=g.ax_marg_y,
        #    color="#7887A4FF",
        #    linewidth=2
        #)

        # 隐藏所有边缘分布的刻度和标签
        g.ax_marg_x.tick_params(
            axis='both',  # 同时操作x和y轴（虽然marg_x只有x轴）
            which='both',  # 主刻度和次刻度
            length=5,  # 刻度线长度为0
            labelbottom=False,  # 隐藏x轴标签
            labelleft=False,
            width=2# 隐藏y轴标签（对marg_x无效，但保留以防万一）
        )
        g.ax_marg_y.tick_params(
            axis='both',
            which='both',
            length=5,
            labelleft=False,  # 隐藏y轴标签
            labelbottom=False,
            width=2
        )


        def get_handles():
            scatter_handle1 = plt.Line2D(
                [0], [0],  # 虚拟数据点
                marker='o',  # 与您的散点图标记一致
                color='w',  # 将线条颜色设为白色（或‘none’）
                markerfacecolor=original_color,  # 标记的填充颜色
                markersize=8,  # 标记大小
                label='Original-Samples',  # 与您的散点图标签一致
                alpha=0.5  # 透明度一致
            )

            scatter_handle2 = plt.Line2D(
                [0], [0],  # 虚拟数据点
                marker='o',  # 与您的散点图标记一致
                color='w',  # 将线条颜色设为白色（或‘none’）
                markerfacecolor=synthetic_color,  # 标记的填充颜色
                markersize=8,  # 标记大小
                label='Synthetic-Samples',  # 与您的散点图标签一致
                alpha=1  # 透明度一致
            )



            # 为柱状图创建一个图例句柄（使用一个矩形补丁）
            bar_handle1 = mpatches.Patch(
                color=original_color,  # 与您的柱状图颜色一致
                alpha=1,  # 与您的柱状图透明度一致
                label='Original-Density'  # 与您的柱状图标签一致
            )

            bar_handle2 = mpatches.Patch(
                color=synthetic_color,  # 与您的柱状图颜色一致
                alpha=0.5,  # 与您的柱状图透明度一致
                label='Synthetic-Density'  # 与您的柱状图标签一致
            )

            # 2. 将句柄和标签合并到一个列表中
            handles = [scatter_handle1, scatter_handle2, bar_handle1, bar_handle2]
            labels = [h.get_label() for h in handles]  # 从句柄中提取对应的标签

            return handles



        g.ax_joint.legend(
            handles = get_handles(),
            loc="lower center",  # 图例位置
            bbox_to_anchor=(0.75, 0.75),
            frameon=False,  # 显示边框
            fontsize=ticksize,  # 字体大小
            edgecolor=None,
            facecolor=None
        )

        g.ax_joint.set_xlim(x_min, x_max)
        g.ax_joint.set_ylim(y_min, y_max)
        g.ax_joint.tick_params(axis='both', labelsize=ticksize)  # 刻度标签字体大小
        g.ax_joint.set_xlabel("Norm. Load", fontsize=labelsize)  # x轴标签字体大小
        g.ax_joint.set_ylabel("Norm. Temp.", fontsize=labelsize)  # y轴标签字体大小
        g.ax_marg_y.set_xlabel('Density', fontsize=labelsize)  # 设置y轴标签
        #g.ax_marg_y.tick_params(axis='y', labelleft=True)  # 确保y轴刻度标签显示
        #g.ax_marg_x.set_xlim(x_min, x_max)
        #g.ax_marg_y.set_ylim(y_min, y_max)
        #g.ax_joint.grid(True, linestyle='-', alpha=0.7, linewidth=0.5)
        #g.fig.subplots_adjust(top=0.8)
        #g.fig.suptitle(country, y=1, fontsize=titlesize)
        plt.locator_params(axis='both', nbins=5)  # 每轴最多显示4个刻度
        plt.tight_layout()
        plt.show()


def plot_generated_label_distribution_coldwave():
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

        sample_load_list = np.mean(load_slice_list[:, -24:], axis=1)
        sample_tem_list = np.mean(tem_slice_list[:, -24:], axis=1)
        generated_sample_load_list = np.mean(coldwave_samples[:, 0, -24:], axis=1)
        generated_sample_tem_list = np.mean(coldwave_samples[:, 1, -24:], axis=1)

        print(sample_load_list.shape)

        df = pd.DataFrame({'Load': sample_load_list, 'Temperature': sample_tem_list,
                           'group': np.repeat('A',sample_load_list.shape[0]) })

        #fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        g = sns.jointplot(data=df, x='Load', y='Temperature',
                          color='gray', label='Original Samples',
                          height=6,  # 控制整个图形的高度
                          ratio=3,  # 主图与边缘图的比例（值越大，主图越小）
                          space=0.2)

        g.ax_joint.scatter(
            generated_sample_load_list, generated_sample_tem_list,
            color="blue", marker="s", alpha=0.7,
            label="Synthetic Samples", s=1
        )

        # 计算统一的坐标范围和分箱
        x_min = 0
        x_max = 0.9
        y_min = 0
        y_max = 0.9
        bins_x = np.linspace(x_min, x_max, 30)
        bins_y = np.linspace(y_min, y_max, 30)
        # 清空原始边缘分布
        g.ax_marg_x.clear()
        g.ax_marg_y.clear()


        # 更新 x 轴边缘分布（包含 new_x）
        # 更新 x 轴边缘分布（包含 new_x）
        # sns.histplot(x=sample_load_list, ax=g.ax_marg_x, bins=bins_x,
        #             color="#682C37FF", alpha=0.5, stat="density")
        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(sample_load_list, bins=bins_x, density=True)
        hist2, _ = np.histogram(generated_sample_load_list, bins=bins_x, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_x.bar(
            x=bins_x[:-1],  # 分箱左边缘作为x坐标
            height=hist1,  # 高度为频数/密度
            width=np.diff(bins_x),  # 宽度为分箱间隔
            color="#682C37FF",  # 原始数据颜色
            alpha=0.5,
            edgecolor="black",
            linewidth=1,
            label="Original"
        )
        g.ax_marg_x.bar(
            x=bins_x[:-1],
            height=hist2,
            width=np.diff(bins_x),
            bottom=hist1,  # 垂直堆叠基准
            color="#A8CDECFF",  # 生成数据颜色
            alpha=1,
            edgecolor="black",
            linewidth=1,
            label="Generated"
        )

        #sns.kdeplot(
        #    x=sample_load_list,
        #    ax=g.ax_marg_x,
        #    color="#682C37FF",
        #    linewidth=2,
        #)

        # sns.histplot(x=generated_sample_load_list, ax=g.ax_marg_x,
        #             color="#A8CDECFF", alpha=1, bins=bins_x, stat="density")
        #sns.kdeplot(
        #    x=generated_sample_load_list,
        #    ax=g.ax_marg_x,
        #    color="#7887A4FF",
        #    linewidth=2,
        #)

        # 更新 y 轴边缘分布（包含 new_y）
        # sns.histplot(y=sample_tem_list, ax=g.ax_marg_y, bins=bins_y,
        #             color="#682C37FF", alpha=0.5, stat="density")
        # sns.histplot(y=generated_sample_tem_list, ax=g.ax_marg_y,
        #             color="#A8CDECFF", alpha=1, bins=bins_y, stat="density")
        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(sample_tem_list, bins=bins_y, density=True)
        hist2, _ = np.histogram(generated_sample_tem_list, bins=bins_y, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist1,
            height=np.diff(bin_edges),
            color="#682C37FF",
            alpha=0.5,
            edgecolor="black",  # 边框颜色
            linewidth=1
        )
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist2,
            height=np.diff(bin_edges),
            left=hist1,  # 关键：以hist1为基准堆叠
            color="#A8CDECFF",
            alpha=1,
            edgecolor="black",  # 边框颜色
            linewidth=1,
        )

        sns.kdeplot(
            y=sample_tem_list,
            ax=g.ax_marg_y,
            color="#682C37FF",
            linewidth=2
        )

        sns.kdeplot(
            y=generated_sample_tem_list,
            ax=g.ax_marg_y,
            color="#7887A4FF",
            linewidth=2
        )

        # 隐藏所有边缘分布的刻度和标签
        g.ax_marg_x.tick_params(
            axis='both',  # 同时操作x和y轴（虽然marg_x只有x轴）
            which='both',  # 主刻度和次刻度
            length=5,  # 刻度线长度为0
            labelbottom=False,  # 隐藏x轴标签
            labelleft=False,
            width=2  # 隐藏y轴标签（对marg_x无效，但保留以防万一）
        )
        g.ax_marg_y.tick_params(
            axis='both',
            which='both',
            length=5,
            labelleft=False,  # 隐藏y轴标签
            labelbottom=False,
            width=2
        )

        g.ax_joint.legend(
            loc="lower left",  # 图例位置
            frameon=True,  # 显示边框
            fontsize=13,  # 字体大小
            edgecolor='black'
        )

        g.ax_joint.set_xlim(x_min, x_max)
        g.ax_joint.set_ylim(y_min, y_max)
        g.ax_joint.tick_params(axis='both', labelsize=14)  # 刻度标签字体大小
        g.ax_joint.set_xlabel("Normalized Load", fontsize=16)  # x轴标签字体大小
        g.ax_joint.set_ylabel("Normalized Temperature", fontsize=16)  # y轴标签字体大小
        #g.ax_marg_x.set_xlim(x_min, x_max)
        #g.ax_marg_y.set_ylim(y_min, y_max)
        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle("Load vs Temperature - Labels", y=1, fontsize=16)
        plt.locator_params(axis='both', nbins=5)  # 每轴最多显示4个刻度
        plt.tight_layout()
        plt.show()


def plot_generated_interval_coldwave(lw=0.2, lw2=1.0, generation=True):
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
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 3), dpi=900)
        # 计算均值、最大值和最小值
        mean_load = np.mean(load_slice_list, axis=0)
        max_load = np.max(load_slice_list, axis=0)
        min_load = np.min(load_slice_list, axis=0)

        mean_tem = np.mean(tem_slice_list, axis=0)
        max_tem = np.max(tem_slice_list, axis=0)
        min_tem = np.min(tem_slice_list, axis=0)
        x = np.arange(192)

        print(mean_load)

        # 绘制均值曲线
        #ax[0].plot(x, mean_curve, color='blue', label='Mean Curve')

        # 绘制包络线
        #line1 = ax[0].fill_between(x, min_load, max_load, color='gray', alpha=0.2, label='Original Samples')
        line1, = ax[0].plot([0], [0], color='gray',
                            label='Common Samples', lw=2)

        line2, = ax[0].plot([0], [0], color='crimson',
                            label='Extreme Samples', lw=2)


        #ax[0].plot(x, coldwave_samples[1, 0, :], color='blue')
        for j in range(0, 50):
            ax[0].plot(x, load_slice_list[j, :], color='gray',
                    lw=lw)
            ax[1].plot(x, tem_slice_list[j, :], color='gray',
                       lw=lw)


        for j in range(1, 2):
            ax[0].plot(x, extreme_load_list[j], color='darkred',
                    lw=lw2)
            ax[1].plot(x, extreme_tem_list[j], color='darkred',
                       lw=lw2)




        #ax[1].fill_between(x, min_tem, max_tem, color='gray', alpha=0.2, label='Envelope')


        if generation:
            for j in range(25, 28):
                ax[0].plot(x, coldwave_samples[j, 0, :], color='darkred',
                           lw=lw2)
                ax[1].plot(x, coldwave_samples[j, 1, :], color='darkred',
                           lw=lw2)


        # 合并两个子图的图例句柄和标签
        handles = [line1, line2]  # 所有线条的句柄
        labels = [h.get_label() for h in handles]  # 对应的标签

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

        ax[0].tick_params(axis='both', labelsize=12)
        ax[1].tick_params(axis='both', labelsize=12)

        #ax[0].spines['top'].set_visible(False)
        #ax[0].spines['right'].set_visible(False)
        #ax[1].spines['top'].set_visible(False)
        #ax[1].spines['right'].set_visible(False)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1, top=0.98)  # 预留底部空间
        #ax[0].set_title('Sample Visualization', fontsize=16)
        plt.show()



#plot_generated_sample_distribution_coldwave(weather_type='coldwave')

#plot_generated_sample_distribution_coldwave(weather_type='common')

#plot_generated_sample_distribution_coldwave(weather_type='noise')

#plot_generated_label_distribution_coldwave()

#plot_generated_interval_coldwave(generation=False)

#plot_generated_interval_coldwave(generation=True, lw2=1)


def plot_generated_sample_distribution_heatwave():
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples_Texas.generate_new_samples_2D import generate_hotwave_samples
    # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe
    for country in ['FAR_WEST']:

        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2019/01/01/00')
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load)) / 1.15
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature)) / 1.1

        T_i_list = np.array([(np.max(temperature[24 * i:24 * (i + 1)]) +
                              np.min(temperature[24 * i:24 * (i + 1)])) / 2
                             for i in range(temperature.shape[0] // 24)])

        T_05 = np.percentile(T_i_list, 5)
        T_95 = np.percentile(T_i_list, 95)
        print(T_05)

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

        coldwave_samples = generate_hotwave_samples(country, num_samples=600).cpu().detach().numpy()
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

        print(sample_load_list.shape)

        df = pd.DataFrame({'Load': sample_load_list, 'Temperature': sample_tem_list,
                           'group': np.repeat('A',sample_load_list.shape[0]) })

        #fig, ax = plt.subplots(2, 1, figsize=(6, 6))

        g = sns.jointplot(data=df, x='Load', y='Temperature',
                          color="#682C37FF", label='Original Samples',
                          height=5,  # 控制整个图形的高度
                          ratio=5,  # 主图与边缘图的比例（值越大，主图越小）
                          space=0.2, s=20, lw=0.1, edgecolor='black', marker='s', alpha=0.5)

        g.ax_joint.scatter(
            generated_sample_load_list, generated_sample_tem_list,
            color="#F6955EFF", marker="^", alpha=1,
            label="Synthetic Samples", s=20, edgecolor='black', lw=0.5
        )

        # 计算统一的坐标范围和分箱
        x_min = 0
        x_max = 0.9
        y_min = 0
        y_max = 0.9
        bins_x = np.linspace(x_min, x_max, 30)
        bins_y = np.linspace(y_min, y_max, 30)
        # 清空原始边缘分布
        g.ax_marg_x.clear()
        g.ax_marg_y.clear()


        # 更新 x 轴边缘分布（包含 new_x）
        #sns.histplot(x=sample_load_list, ax=g.ax_marg_x, bins=bins_x,
        #             color="#682C37FF", alpha=0.5, stat="density")
        # sns.histplot(x=generated_sample_load_list, ax=g.ax_marg_x,
        #             color="#A8CDECFF", alpha=0.7, bins=bins_x, stat="density")

        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(sample_load_list, bins=bins_x, density=True)
        hist2, _ = np.histogram(generated_sample_load_list, bins=bins_x, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_x.bar(
            x=bins_x[:-1],  # 分箱左边缘作为x坐标
            height=hist1,  # 高度为频数/密度
            width=np.diff(bins_x),  # 宽度为分箱间隔
            color="#682C37FF",  # 原始数据颜色
            alpha=0.5,
            edgecolor="black",
            linewidth=1,
            label="Original"
        )
        g.ax_marg_x.bar(
            x=bins_x[:-1],
            height=hist2,
            width=np.diff(bins_x),
            bottom=hist1,  # 垂直堆叠基准
            color="#F6955EFF",  # 生成数据颜色
            alpha=1,
            edgecolor="black",
            linewidth=1,
            label="Generated"
        )



        sns.kdeplot(
            x=sample_load_list,
            ax=g.ax_marg_x,
            color="#682C37FF",
            linewidth=2,
        )


        sns.kdeplot(
            x=generated_sample_load_list,
            ax=g.ax_marg_x,
            color="#F6955EFF",
            linewidth=2,
        )


        # 更新 y 轴边缘分布（包含 new_y）
        #sns.histplot(y=sample_tem_list, ax=g.ax_marg_y, bins=bins_y,
        #             color="#682C37FF", alpha=0.5, stat="density")
        #sns.histplot(y=generated_sample_tem_list, ax=g.ax_marg_y,
        #             color="#A8CDECFF", alpha=0.8, bins=bins_y, stat="density")

        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(sample_tem_list, bins=bins_y, density=True)
        hist2, _ = np.histogram(generated_sample_tem_list, bins=bins_y, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist1,
            height=np.diff(bin_edges),
            color="#682C37FF",
            alpha=0.5,
            edgecolor="black",  # 边框颜色
            linewidth=1
        )
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist2,
            height=np.diff(bin_edges),
            left=hist1,  # 关键：以hist1为基准堆叠
            color="#F6955EFF",
            alpha=1,
            edgecolor="black",  # 边框颜色
            linewidth=1,
        )

        sns.kdeplot(
            y=sample_tem_list,
            ax=g.ax_marg_y,
            color="#682C37FF",
            linewidth=2
        )

        sns.kdeplot(
            y=generated_sample_tem_list,
            ax=g.ax_marg_y,
            color="#F6955EFF",
            linewidth=2
        )

        # 隐藏所有边缘分布的刻度和标签
        g.ax_marg_x.tick_params(
            axis='both',  # 同时操作x和y轴（虽然marg_x只有x轴）
            which='both',  # 主刻度和次刻度
            length=5,  # 刻度线长度为0
            labelbottom=False,  # 隐藏x轴标签
            labelleft=False,
            width=2# 隐藏y轴标签（对marg_x无效，但保留以防万一）
        )
        g.ax_marg_y.tick_params(
            axis='both',
            which='both',
            length=5,
            labelleft=False,  # 隐藏y轴标签
            labelbottom=False,
            width=2
        )

        g.ax_joint.legend(
            loc="lower right",  # 图例位置
            frameon=True,  # 显示边框
            fontsize=12,  # 字体大小
            edgecolor='black'
        )


        g.ax_joint.set_xlim(x_min, x_max)
        g.ax_joint.set_ylim(y_min, y_max)
        g.ax_joint.tick_params(axis='both', labelsize=12)  # 刻度标签字体大小
        g.ax_joint.set_xlabel("Norm. Load", fontsize=15)  # x轴标签字体大小
        g.ax_joint.set_ylabel("Norm. Temp.", fontsize=15)  # y轴标签字体大小
        #g.ax_marg_x.set_xlim(x_min, x_max)
        #g.ax_marg_y.set_ylim(y_min, y_max)
        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle(country, y=1, fontsize=16)
        plt.locator_params(axis='both', nbins=5)  # 每轴最多显示4个刻度
        plt.tight_layout()
        plt.show()


def plot_generated_label_distribution_heatwave():
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples_Texas.generate_new_samples_2D import generate_hotwave_samples
    # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe
    for country in ['NORTH']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2019/01/01/00')
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load)) / 1.15
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature)) / 1.1

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

        coldwave_samples = generate_hotwave_samples(country, num_samples=600).cpu().detach().numpy()
        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)


        load_slice_list = np.array(load_slice_list)
        #print(load_slice_list.shape)
        tem_slice_list = np.array(tem_slice_list)

        sample_load_list = np.mean(load_slice_list[:, -24:], axis=1)
        sample_tem_list = np.mean(tem_slice_list[:, -24:], axis=1)
        generated_sample_load_list = np.mean(coldwave_samples[:, 0, -24:], axis=1)
        generated_sample_tem_list = np.mean(coldwave_samples[:, 1, -24:], axis=1)

        print(sample_load_list.shape)

        df = pd.DataFrame({'Load': sample_load_list, 'Temperature': sample_tem_list,
                           'group': np.repeat('A',sample_load_list.shape[0]) })

        #fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        g = sns.jointplot(data=df, x='Load', y='Temperature',
                          color="#682C37FF", label='Original Samples',
                          height=6,  # 控制整个图形的高度
                          ratio=3,  # 主图与边缘图的比例（值越大，主图越小）
                          space=0.2)

        g.ax_joint.scatter(
            generated_sample_load_list, generated_sample_tem_list,
            color="#F6955EFF", marker="s", alpha=0.7, s=5,
            label="Synthetic Samples"
        )

        # 计算统一的坐标范围和分箱
        x_min = 0
        x_max = 0.9
        y_min = 0
        y_max = 0.9
        bins_x = np.linspace(x_min, x_max, 30)
        bins_y = np.linspace(y_min, y_max, 30)
        # 清空原始边缘分布
        g.ax_marg_x.clear()
        g.ax_marg_y.clear()


        # 更新 x 轴边缘分布（包含 new_x）
        # sns.histplot(x=sample_load_list, ax=g.ax_marg_x, bins=bins_x,
        #             color="#682C37FF", alpha=0.5, stat="density")
        # sns.histplot(x=generated_sample_load_list, ax=g.ax_marg_x,
        #             color="#A8CDECFF", alpha=0.7, bins=bins_x, stat="density")

        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(sample_load_list, bins=bins_x, density=True)
        hist2, _ = np.histogram(generated_sample_load_list, bins=bins_x, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_x.bar(
            x=bins_x[:-1],  # 分箱左边缘作为x坐标
            height=hist1,  # 高度为频数/密度
            width=np.diff(bins_x),  # 宽度为分箱间隔
            color="#682C37FF",  # 原始数据颜色
            alpha=0.5,
            edgecolor="black",
            linewidth=1,
            label="Original"
        )
        g.ax_marg_x.bar(
            x=bins_x[:-1],
            height=hist2,
            width=np.diff(bins_x),
            bottom=hist1,  # 垂直堆叠基准
            color="#F6955EFF",  # 生成数据颜色
            alpha=1,
            edgecolor="black",
            linewidth=1,
            label="Generated"
        )

        sns.kdeplot(
            x=sample_load_list,
            ax=g.ax_marg_x,
            color="#682C37FF",
            linewidth=2,
        )

        sns.kdeplot(
            x=generated_sample_load_list,
            ax=g.ax_marg_x,
            color="#F6955EFF",
            linewidth=2,
        )

        # 更新 y 轴边缘分布（包含 new_y）
        # sns.histplot(y=sample_tem_list, ax=g.ax_marg_y, bins=bins_y,
        #             color="#682C37FF", alpha=0.5, stat="density")
        # sns.histplot(y=generated_sample_tem_list, ax=g.ax_marg_y,
        #             color="#A8CDECFF", alpha=0.8, bins=bins_y, stat="density")

        # 计算两组数据的直方图
        hist1, bin_edges = np.histogram(sample_tem_list, bins=bins_y, density=True)
        hist2, _ = np.histogram(generated_sample_tem_list, bins=bins_y, density=True)

        # 绘制堆叠柱状图
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist1,
            height=np.diff(bin_edges),
            color="#682C37FF",
            alpha=0.5,
            edgecolor="black",  # 边框颜色
            linewidth=1
        )
        g.ax_marg_y.barh(
            y=bin_edges[:-1],
            width=hist2,
            height=np.diff(bin_edges),
            left=hist1,  # 关键：以hist1为基准堆叠
            color="#F6955EFF",
            alpha=1,
            edgecolor="black",  # 边框颜色
            linewidth=1,
        )

        sns.kdeplot(
            y=sample_tem_list,
            ax=g.ax_marg_y,
            color="#682C37FF",
            linewidth=2
        )

        sns.kdeplot(
            y=generated_sample_tem_list,
            ax=g.ax_marg_y,
            color="#F6955EFF",
            linewidth=2
        )

        # 隐藏所有边缘分布的刻度和标签
        g.ax_marg_x.tick_params(
            axis='both',  # 同时操作x和y轴（虽然marg_x只有x轴）
            which='both',  # 主刻度和次刻度
            length=5,  # 刻度线长度为0
            labelbottom=False,  # 隐藏x轴标签
            labelleft=False,
            width=2  # 隐藏y轴标签（对marg_x无效，但保留以防万一）
        )
        g.ax_marg_y.tick_params(
            axis='both',
            which='both',
            length=5,
            labelleft=False,  # 隐藏y轴标签
            labelbottom=False,
            width=2
        )

        g.ax_joint.legend(
            loc="lower right",  # 图例位置
            frameon=True,  # 显示边框
            fontsize=13,  # 字体大小
            edgecolor='black'
        )

        g.ax_joint.set_xlim(x_min, x_max)
        g.ax_joint.set_ylim(y_min, y_max)
        g.ax_joint.tick_params(axis='both', labelsize=14)  # 刻度标签字体大小
        g.ax_joint.set_xlabel("Normalized Load", fontsize=16)  # x轴标签字体大小
        g.ax_joint.set_ylabel("Normalized Temperature", fontsize=16)  # y轴标签字体大小
        #g.ax_marg_x.set_xlim(x_min, x_max)
        #g.ax_marg_y.set_ylim(y_min, y_max)
        g.fig.subplots_adjust(top=0.8)
        g.fig.suptitle("Load vs Temperature - Labels", y=1, fontsize=16)
        plt.locator_params(axis='both', nbins=5)  # 每轴最多显示4个刻度
        plt.tight_layout()
        plt.show()


def plot_generated_interval_heatwave():
    import sys
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/Model_parameters')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/diff_training_2D')
    sys.path.append('/home/ln/workspace/ExtremeWeather/forecasting_using_generated_samples_Texas/diff_Model_2D')
    print(sys.path)
    from forecasting_using_generated_samples.generate_new_samples_2D import generate_hotwave_samples
    # from forecasting_using_generated_samples.Forecasting_model_training import generate_coldwave_samples
    # Europe
    for country in ['COAST']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/Texas_reformed_data/{}.xlsx'.format(country))
        start_date = pd.to_datetime('2019/01/01/00')
        end_date = pd.to_datetime('2022/12/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        load = np.array(data['Load'])
        load = (load - min(load)) / (max(load) - min(load)) / 1.15
        temperature = np.array(data['Temperature'])
        temperature = (temperature - min(temperature)) / (max(temperature) - min(temperature)) / 1.1

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


        coldwave_samples = generate_hotwave_samples(country, num_samples=50).cpu().detach().numpy()
        coldwave_samples = coldwave_samples.reshape(coldwave_samples.shape[0],
                                                    coldwave_samples.shape[1], -1)

        load_slice_list = np.array(load_slice_list)
        print(load_slice_list.shape)
        tem_slice_list = np.array(tem_slice_list)

        print(load_slice_list)


        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        # 计算均值、最大值和最小值
        mean_load = np.mean(load_slice_list, axis=0)
        max_load = np.max(load_slice_list, axis=0)
        min_load = np.min(load_slice_list, axis=0)

        mean_tem = np.mean(tem_slice_list, axis=0)
        max_tem = np.max(tem_slice_list, axis=0)
        min_tem = np.min(tem_slice_list, axis=0)
        x = np.arange(192)

        print(mean_load)

        # 绘制均值曲线
        #ax[0].plot(x, mean_curve, color='blue', label='Mean Curve')

        # 绘制包络线
        line1 = ax[0].fill_between(x, min_load, max_load, color='gray', alpha=0.2, label='Original Samples')
        line2, = ax[0].plot(x, coldwave_samples[0, 0, :], color='red', label='Synthetic Samples')
        ax[0].plot(x, coldwave_samples[15, 0, :], color='red')
        ax[0].plot(x, coldwave_samples[25, 0, :], color='red')
        ax[0].plot(x, coldwave_samples[45, 0, :], color='red')


        ax[1].fill_between(x, min_tem, max_tem, color='gray', alpha=0.2)
        ax[1].plot(x, coldwave_samples[0, 1, :], color='red')
        ax[1].plot(x, coldwave_samples[15, 1, :], color='red')
        ax[1].plot(x, coldwave_samples[25, 1, :], color='red')
        ax[1].plot(x, coldwave_samples[45, 1, :], color='red')

        # 合并两个子图的图例句柄和标签
        handles = [line1, line2]  # 所有线条的句柄
        labels = [h.get_label() for h in handles]  # 对应的标签

        # 在图像底部添加共同图例
        fig.legend(
            handles=handles,
            labels=labels,
            loc="lower center",  # 图例位置（底部居中）
            bbox_to_anchor=(0.5, 0),  # 调整位置（x, y偏移）
            ncol=2,  # 图例列数（根据条目数量调整）
            frameon=True,  # 是否显示边框
            fontsize=15,
            edgecolor='black'
        )

        # 创建大括号路径
        def square_bracket(x, y, width, height):
            """生成精确的 ']' 形路径"""
            verts = [
                # 右侧垂直线（从上到下）
                (x + width, y + height),  # 起点：右上角
                (x + width, y - height),  # 右下角

                # 底部小横线（右到左）
                (x + 0.8 * width, y - height),  # 控制点1
                (x + 0.6 * width, y - height),  # 控制点2

                # 左侧垂直线（下到上）
                (x, y - height),  # 左下角
                (x, y + height),  # 左上角

                # 顶部小横线（左到右）
                #(x + 0.6 * width, y + height),  # 控制点1
                #(x + 0.8 * width, y + height),  # 控制点2

                # 闭合路径
                #(x + width, y + height)  # 回到起点
            ]
            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.CURVE3,  # 二次贝塞尔曲线控制点
                Path.CURVE3,
                Path.LINETO,
                Path.LINETO
            ]
            return Path(verts, codes)

        # 添加到图形
        brace = PathPatch(
            square_bracket(169, 0.2, 23, 0.02),  # 参数：x中心, y中心, 宽度, 高度
            facecolor='none',
            edgecolor='black',
            linewidth=2,
            linestyle='-'
        )

        ax[0].add_patch(brace)

        ax[0].text(
            x=168+12,  # 文字位置的x坐标（数据坐标系）
            y=0.1,  # 文字位置的y坐标
            s="Label",  # 文字内容
            fontsize=14,  # 字体大小
            color="black",  # 颜色
            ha="center",  # 水平对齐：'left', 'center', 'right'
            va="center"  # 垂直对齐：'top', 'center', 'bottom'
        )

        ax[0].set_ylabel('Normalized Load', fontsize=14)
        ax[1].set_ylabel('Normalized Temperature', fontsize=14)
        ax[1].set_xlabel('Time Index [h]', fontsize=14)

        ax[0].tick_params(axis='both', labelsize=14)
        ax[1].tick_params(axis='both', labelsize=14)

        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2, top=0.9)  # 预留底部空间
        ax[0].set_title('Sample Visualization', fontsize=16)
        plt.show()


#plot_generated_sample_distribution_heatwave()

#plot_generated_label_distribution_heatwave()

#plot_generated_interval_heatwave()




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


def basic_figure_multi(sheet_name_list, title_list, metric='MAE'):
    font_size = 18
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
    color_list = ["#A8CDECFF", "#FDDED7", "#C1E0DB"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
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
        bar_width = 0.23  # 减小柱子宽度
        gap = 0.08  # 新增：组内柱子间隔

        ax = axes[idx]
        for i in range(len(Setting)):
            # 调整x坐标：增加间隔 (gap)
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
        total_width = len(Setting) * bar_width + (len(Setting) - 1) * gap
        center = x + total_width / 2 - bar_width / 2
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
    '2018 Europe Coldwave',
    '2023 Texas Heatwave',
    '2023 PJM Heatwave',
    '2022 India Heatwave',
    '2022 Guangdong Heatwave',
    '2023 Hunan Coldwave'
]

#basic_figure_multi(sheet_name_list, title_list)
#ablation_figure_multi(sheet_name_list, title_list, metric='RMSE')


def convergence():
    MLP_Hunan = []
    LSTM_Hunan = []
    CNN_Hunan = []
    test_country_list = ['娄底', '岳阳',
                         '常德',
                         '张家界',
                         '怀化',
                         '株洲',
                         '永州',
                         '湘潭',
                         '湘西',
                         '益阳',
                         '衡阳',
                         '邵阳',
                         '郴州',
                         '长沙']
    for country in test_country_list:
        data = pd.read_excel(
            '../forecasting_using_generated_samples_hunan/Convergence_curve/convergence_{}_ANN.xlsx'.format(
                country)).values[:100, 1]
        #print(data)
        MLP_Hunan.append(data)

        data = pd.read_excel(
            '../forecasting_using_generated_samples_hunan/Convergence_curve/convergence_{}_LSTM.xlsx'.format(
                country)).values[:100, 1]
        LSTM_Hunan.append(data)

        data = pd.read_excel(
            '../forecasting_using_generated_samples_hunan/Convergence_curve/convergence_{}_CNN.xlsx'.format(
                country)).values[:100, 1]
        CNN_Hunan.append(data)

    MLP_Hunan = np.array(MLP_Hunan)
    LSTM_Hunan = np.array(LSTM_Hunan)
    CNN_Hunan = np.array(CNN_Hunan)

    mean_curve_MLP_Hunan = np.mean(MLP_Hunan, axis=0)
    max_curve_MLP_Hunan = np.max(MLP_Hunan, axis=0)
    min_curve_MLP_Hunan = np.min(MLP_Hunan, axis=0)


    MLP_PJM = []
    LSTM_PJM = []
    CNN_PJM = []
    test_country_list = ['Allegheny Power System',
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
    for country in test_country_list:
        data = pd.read_excel(
            '../forecasting_using_generated_samples_PJM/Convergence_curve/convergence_{}_ANN.xlsx'.format(
                country)).values[:100, 1]
        # print(data)
        MLP_PJM.append(data)

        data = pd.read_excel(
            '../forecasting_using_generated_samples_PJM/Convergence_curve/convergence_{}_LSTM.xlsx'.format(
                country)).values[:100, 1]
        LSTM_PJM.append(data)

        data = pd.read_excel(
            '../forecasting_using_generated_samples_PJM/Convergence_curve/convergence_{}_CNN.xlsx'.format(
                country)).values[:100, 1]
        CNN_PJM.append(data)


    MLP_PJM = np.array(MLP_PJM)
    LSTM_PJM = np.array(LSTM_PJM)
    CNN_PJM = np.array(CNN_PJM)

    mean_curve_MLP_PJM = np.mean(MLP_PJM, axis=0)
    max_curve_MLP_PJM = np.max(MLP_PJM, axis=0)
    min_curve_MLP_PJM = np.min(MLP_PJM, axis=0)
    x = np.linspace(0, 100, 100)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.fill_between(x, min_curve_MLP_Hunan, max_curve_MLP_Hunan, color='gray', alpha=0.2)
    ax.plot(x, mean_curve_MLP_Hunan, color='red')
    ax.fill_between(x, min_curve_MLP_PJM, max_curve_MLP_PJM, color='gray', alpha=0.2)
    ax.plot(x, mean_curve_MLP_PJM, color='blue')
    ax.set_ylim(ymin=0, ymax=0.3)

    plt.show()

#convergence()



def data_scaler(country='Belgium', strat_time = '2021/03/28/00', end_time = '2023/05/31/23'):
    data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(country), header=0,
                       usecols=['Date_Hour', 'Load', 'Temperature'])

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

    return max(load), min(load), max(temperature), min(temperature)



def plot_heatwave_curve():
    #plt.rcParams['xtick.labelsize'] = 12  # x轴刻度字体
    #plt.rcParams['ytick.labelsize'] = 12
    for country in ['Pennsylvania Electric Company']:
        # country = 'COAST'
        data = pd.read_csv('../Data/reformed_data_updated/PJM_reformed_data/{}.csv'.format(country), header=0,
                           usecols=['Date_Hour', 'Load', 'Temperature'])

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

        start_date = pd.to_datetime('2022/10/01/00')
        end_date = pd.to_datetime('2023/07/20/23')
        data = data[(pd.to_datetime(data['Date_Hour']) >= start_date) & (pd.to_datetime(data['Date_Hour']) <= end_date)]

        maxload, minload, maxtem, mintem = data_scaler(country)

        load = np.array(data['Load'])
        load = (load - minload) / (maxload - minload)
        temperature = np.array(data['Temperature'])
        temperature = (temperature - mintem) / (maxtem - mintem)

        tem = [(np.max(temperature[24*i:24*(i+1)])+
                np.min(temperature[24*i:24*(i+1)]))/2 for i in range(temperature.shape[0]//24)]

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))

        x1 = np.linspace(0, load.shape[0], load.shape[0])
        ax1.plot(x1, load, label='Load', color='#A8CDECFF', lw=0.5)
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Time Index [h]')
        ax1.set_ylabel('Norm. Load')
        #ax1.tick_params(axis='y', labelcolor='#A8CDECFF')


        x2 = np.linspace(0, load.shape[0], load.shape[0]//24)
        ax2 = ax1.twinx()

        ax2.plot(x2, tem, label='Temperature', color='#9B6981FF', lw=0.5)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Norm. Daily Temp.')

        ax1.fill_between([temperature.shape[0]-600+i for i in range(600)],  # x值范围
                        [0 for i in range(600)],  # 下边界（曲线本身）
                        [1 for i in range(600)],  # 上边界（比曲线最高点高2个单位）
                        color='salmon',
                        alpha=0.2)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_color('#9B6981FF')  # 轴线颜色
        ax2.yaxis.label.set_color('#9B6981FF')  # 标签颜色
        ax2.tick_params(axis='y', colors='#9B6981FF')  # 刻度颜色

        ax1.annotate('Coldwave',
                     xy=(6400, 0.82),  # 箭头终点位置
                     xytext=(5400, 0.82),  # 箭头起点位置（向右偏移300单位）
                     arrowprops=dict(arrowstyle='->', color='crimson', lw=2),
                     ha='center')

        plt.title('Load Visualization - Heatwaves')
        plt.tight_layout()
        #ax2.spines['right'].set_visible(False)

        plt.show()




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



def plot_coldwave_curve():
    #plt.rcParams['xtick.labelsize'] = 12  # x轴刻度字体
    #plt.rcParams['ytick.labelsize'] = 12
    for country in ['France']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))

        start_date = pd.to_datetime('2017/09/01/00')
        end_date = pd.to_datetime('2018/05/30/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        maxload, minload, maxtem, mintem = coldwave_scaler(country)

        load = np.array(data['Load'])
        load = (load - minload) / (maxload - minload)
        temperature = np.array(data['Temperature'])
        temperature = (temperature - mintem) / (maxtem - mintem)

        tem = [(np.max(temperature[24*i:24*(i+1)])+
                np.min(temperature[24*i:24*(i+1)]))/2 for i in range(temperature.shape[0]//24)]

        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3.5))

        x1 = np.linspace(0, load.shape[0], load.shape[0])
        line1, = ax1.plot(x1, load, color='#46AEA0FF', lw=0.15)
        line1, = ax1.plot(x1[0], load[0], label='Load', color='#88A0DCFF', lw=1)
        ax1.set_ylim([-0.1, 1.2])
        ax1.set_xlabel('Time Index [h]')
        ax1.set_ylabel('Norm. Load', fontsize=14)
        #ax1.tick_params(axis='y', labelcolor='#A8CDECFF')


        x2 = np.linspace(0, load.shape[0], load.shape[0]//24)
        ax2 = ax1.twinx()

        line2, = ax2.plot(x2, tem, label='Daily Temp.', color='#C24841FF', lw=0.8)
        ax2.set_ylim([-0.1, 1.2])
        ax2.set_ylabel('Norm. Daily Temp.', fontsize=14)

        line3 = ax1.fill_between([4100+i for i in range(300)],  # x值范围
                        [0 for i in range(300)],  # 下边界（曲线本身）
                        [1.05 for i in range(300)],  # 上边界（比曲线最高点高2个单位）
                        color='blue',
                        alpha=0.1, label='Coldwave Period')

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        #ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_color('#C24841FF')  # 轴线颜色
        ax2.yaxis.label.set_color('#C24841FF')  # 标签颜色
        ax2.tick_params(axis='y', colors='#C24841FF')  # 刻度颜色

        lines = [line1, line2, line3]  # 手动组合线条对象
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left', edgecolor='black', frameon=False)

        #ax1.annotate('Coldwave',
        #             xy=(4500, 0.8),  # 箭头终点位置
        #             xytext=(5500, 0.8),  # 箭头起点位置（向右偏移300单位）
        #             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
        #             ha='center')


        #plt.title('Load Visualization - Coldwaves')
        ax1.set_xticks([0, 2000, 4000, 6000])
        ax1.set_yticks([0, 0.4, 0.8, 1.2])
        ax2.set_yticks([0, 0.4, 0.8, 1.2])
        ax1.tick_params(axis='both', labelsize=14)  # 同时设置x轴和y轴刻度标签
        ax2.tick_params(axis='both', labelsize=14)  # 同时设置x轴和y轴刻度标签

        ax1.margins(x=0)
        ax2.margins(x=0)
        plt.tight_layout()
        #ax2.spines['right'].set_visible(False)

        plt.show()


def plot_coldwave_curve_2(titlesize=18, ticksize=15, labelsize=18):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    for country in ['France']:
        # country = 'COAST'
        data = pd.read_excel('../Data/reformed_data_updated/real_Europe_data_reformed/{}.xlsx'.format(country))

        start_date = pd.to_datetime('2017/08/01/00')
        end_date = pd.to_datetime('2018/07/31/23')
        data = data[(pd.to_datetime(data['Data_Hour']) >= start_date) & (pd.to_datetime(data['Data_Hour']) <= end_date)]

        maxload, minload, maxtem, mintem = coldwave_scaler(country)

        load = np.array(data['Load'])
        load = (load - minload) / (maxload - minload)
        temperature = np.array(data['Temperature'])
        temperature = (temperature - mintem) / (maxtem - mintem)

        tem = [(np.max(temperature[24*i:24*(i+1)])+
                np.min(temperature[24*i:24*(i+1)]))/2 for i in range(temperature.shape[0]//24)]

        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        ax1 = ax[0]
        ax2 = ax[1]

        x1 = np.linspace(0, load.shape[0], load.shape[0])
        line1, = ax1.plot(x1, load, color='#5480B5FF', lw=0.15)
        line1, = ax1.plot(x1[0], load[0], label='Load', color='#88A0DCFF', lw=1)
        ax1.set_ylim([0, 1.1])
        #ax1.set_xlabel('Time Index [h]')
        ax1.set_ylabel('Norm. Load', fontsize=labelsize, labelpad=10)
        #ax1.tick_params(axis='y', labelcolor='#A8CDECFF')
        ax1.set_title('Historical Load and Temp. Data', fontsize=titlesize)


        x2 = np.linspace(0, load.shape[0], load.shape[0]//24)
        #ax2 = ax1.twinx()

        line2, = ax2.plot(x2, tem, label='Daily Temp.', color='#C24841FF', lw=0.8)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Norm. Daily Temp.', fontsize=labelsize, labelpad=10)

        line3 = ax1.fill_between([4100+i+31*24 for i in range(300)],  # x值范围
                        [0 for i in range(300)],  # 下边界（曲线本身）
                        [1.05 for i in range(300)],  # 上边界（比曲线最高点高2个单位）
                        color='blue',
                        alpha=0.1, label='Coldwave')

        ax2.fill_between([4100 + i + 31 * 24 for i in range(300)],  # x值范围
                         [0 for i in range(300)],  # 下边界（曲线本身）
                         [1.05 for i in range(300)],  # 上边界（比曲线最高点高2个单位）
                         color='blue',
                         alpha=0.1, label='Coldwave')

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        #ax2.spines['right'].set_color('#C24841FF')  # 轴线颜色
        #ax2.yaxis.label.set_color('#C24841FF')  # 标签颜色
        #ax2.tick_params(axis='y', colors='#C24841FF')  # 刻度颜色
        #ax1.grid()

        lines = [line1, line2, line3]  # 手动组合线条对象
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels,
                   loc='upper left', edgecolor='black',
                   frameon=False, ncols=1,
                   handlelength=1, columnspacing=1, fontsize=ticksize)

        #ax1.annotate('Coldwave',
        #             xy=(4500, 0.8),  # 箭头终点位置
        #             xytext=(5500, 0.8),  # 箭头起点位置（向右偏移300单位）
        #             arrowprops=dict(arrowstyle='->', color='blue', lw=2),
        #             ha='center')


        #plt.title('Load Visualization - Coldwaves')
        #ax1.set_xticks([0, 2000, 4000, 6000, 8000])
        ax1.set_yticks([0, 0.4, 0.8])
        x = [31*24, 123*24, 215*24, 304*24]
        labels = ['Sep.', 'Dec.', 'Mar.', 'Jun.']
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=0, ha='center')  # 旋转45度



        ax2.set_yticks([0, 0.4, 0.8])
        ax1.tick_params(axis='both', labelsize=ticksize)  # 同时设置x轴和y轴刻度标签
        ax2.tick_params(axis='both', labelsize=ticksize)  # 同时设置x轴和y轴刻度标签

        ax1.margins(x=0)
        ax2.margins(x=0)
        plt.tight_layout()
        #ax2.spines['right'].set_visible(False)

        plt.show()

#plot_heatwave_curve()

#plot_coldwave_curve()

#plot_coldwave_curve_2()

#plot_heatwave_curve_2()


def separation_convergence():
    countries = ['Europe', 'Guangdong', 'hunan', 'India', 'PJM', 'Texas']

    regions = [
        ['chaozhou', 'dongguan', 'foshan', 'guangzhou', 'heyuan', 'huizhou',
                   'jiangmen', 'jieyang', 'maoming', 'meizhou', 'qingyuan', 'shantou',
                   'shanwei', 'shaoguan', 'shenzhen', 'yangjiang', 'yunfu', 'zhanjiang',
                   'zhaoqing', 'zhongshan', 'zhuhai']]

    ANN_loss_list = []
    CNN_loss_list = []
    LSTM_loss_list = []

    for country in countries:
        ANN_loss = []
        CNN_loss = []
        LSTM_loss = []



def plot_map():
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path

    # 获取广东省数据
    guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/440000_full.json')
    hunan = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/430000_full.json')

    states = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip")
    texas = states[states["name"] == "Texas"]

    # 获取世界地图背景
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 8))
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)  # 背景
    guangdong.plot(ax=ax, color='blue', edgecolor='blue', linewidth=0.8)  # 广东省
    hunan.plot(ax=ax, color='green', edgecolor='green', linewidth=0.8)
    texas.plot(ax=ax, color='red', edgecolor='green', linewidth=0.8)

    # 设置标题和显示
    plt.title("Guangdong Province (Blue) on World Map", fontsize=14)
    plt.tight_layout()
    plt.show()



def plot_guangdong_map(titlesize=16, ticksize=14, labelsize=14):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path



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
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plt.axis('equal')
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)


    cmap = plt.get_cmap('GnBu')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(1, 1.5)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
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

            if ECF == 0 and EHF == 0:
                common_load_mean.append(load[24 * j: 24 * (j + 1)])
                common_load_norm.append(norm_load[24 * j: 24 * (j + 1)])


        hot_common_ratio = np.mean(hotwave_load_mean)/np.mean(common_load_mean)
        color = cmap(norm(hot_common_ratio))

        print(i)
        guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        #loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.55, 0.15, 0.4, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(hotwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')



    ax.set_yticks([20, 22, 24, 26])
    ax.set_xticks([110, 112, 114, 116, 118])
    ax.set_xticklabels(['110°E', '112°E', '114°E', '116°E', '118°E'])
    ax.set_yticklabels(['20°N', '22°N', '24°N', '26°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Patterns', fontsize=labelsize)
    ax_inset.set_xlabel('Hour', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_xlim(109, 119)
    ax.set_ylim(19, 26)
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(1)

    # 设置标题和显示
    ax.set_title("Heatwave in Guangdong", fontsize=titlesize)
    plt.tight_layout()
    #plt.axis('equal')
    plt.show()


def plot_hunan_map(titlesize=16, ticksize=14, labelsize=14):
    import pandas as pd
    import geopandas
    import matplotlib.pyplot as plt
    from geodatasets import get_path

    city_name = ['娄底', '岳阳', '常德', '张家界', '怀化', '株洲',
                 '永州', '湘潭', '湘西', '益阳', '衡阳', '邵阳', '郴州', '长沙']

    city_code = ['431300', '430600', '430700', '430800', '431200',
                 '430200', '431100', '430300', '433100', '430900',
                 '430400', '430500', '431000', '430100']

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    world = gpd.read_file("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip")
    world.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)


    cmap = plt.get_cmap('RdPu')  # 黄-橙-红色标，适用于热力值
    norm = Normalize(1, 1.5)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
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


        hot_common_ratio = np.mean(hotwave_load_mean)/np.mean(common_load_mean)
        color = cmap(norm(hot_common_ratio))

        print(i)
        guangdong = gpd.read_file('https://geo.datav.aliyun.com/areas_v3/bound/geojson?code={}'.format(city_code[i]))

        guangdong.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        #loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.55, 0.15, 0.4, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(hotwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')



    ax.set_yticks([24, 26, 28, 30, 32, 34])
    ax.set_xticks([108, 110, 112, 114, 116, 118])
    ax.set_xticklabels(['108°E', '110°E', '112°E', '114°E', '116°E', '118°E'])
    ax.set_yticklabels(['24°N', '26°N', '28°N', '30°N', '32°N', '34°N'])

    ax_inset.set_yticks([0.5, 1])
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.xaxis.set_tick_params(labelsize=ticksize)
    ax_inset.yaxis.set_tick_params(labelsize=ticksize)
    ax_inset.set_title('Load Patterns', fontsize=labelsize)
    ax_inset.set_xlabel('Hour', fontsize=labelsize)
    ax_inset.set_ylabel('Load', fontsize=labelsize)

    ax.set_xlim(108, 118)
    ax.set_ylim(23, 30)
    ax.set_aspect(1, adjustable='datalim')
    ax.set_box_aspect(1)

    # 设置标题和显示
    ax.set_title("Coldwave in Hunan", fontsize=titlesize)
    plt.tight_layout()
    #plt.axis('equal')
    plt.show()


def plot_europe_map(titlesize=16, ticksize=14, labelsize=14):
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
    norm = Normalize(1, 1.5)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 必须设置一个空数组
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.7, pad=0.1, aspect=30)
    cbar.set_label('Load Increase Ratio in Heatwave Periods', fontsize=labelsize)
    cbar.ax.tick_params(labelsize=ticksize)  # 调整色标字体大小

    common_load_norm = []
    hotwave_load_norm = []
    coldwave_load_norm = []
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


        hot_common_ratio = np.mean(coldwave_load_mean)/np.mean(common_load_mean)
        color = cmap(norm(hot_common_ratio))

        print(i)
        target = world[world['NAME'].str.contains(country, case=False)]
        #target = world[world['NAME'].str.contains(country, case=False)]
        #print(target)

        target.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8)  # 广东省

    ax_inset = inset_axes(
        ax,
        width="100%", height="100%",
        #loc="lower right",  # 小图的锚点位置
        bbox_to_anchor=(0.55, 0.15, 0.4, 0.2),  # 锚点偏移（x=1.05 表示主图右侧外）
        bbox_transform=ax.transAxes,  # 使用主图坐标系
        borderpad=0
    )

    ax_inset.plot(np.mean(np.array(coldwave_load_norm), axis=0), marker='s', markevery=2,
                      markerfacecolor='white', color='crimson')

    ax_inset.plot(np.mean(np.array(common_load_norm), axis=0), marker='o', markevery=2,
                      markerfacecolor='white', color='black')



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
    ax.set_box_aspect(1)

    # 设置标题和显示
    ax.set_title("Coldwave in Europe", fontsize=titlesize)
    plt.tight_layout()
    #plt.axis('equal')
    plt.show()


#plot_map()

#plot_guangdong_map()

#plot_hunan_map()

#plot_europe_map()

#def calculate_dif(x1, x2):

#    print(round((x1-x2)/x1*100, 2))

#calculate_dif(0.04953,0.02881)
