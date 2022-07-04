import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from d_2021.config.path import *
from d_2021.utils import common_utils
from d_2021.utils.common_utils import list_to_float_df, get_series_name
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def get_box_plot(m_list: list, name, show_plt):
    """
    绘制箱型图
    :param show_plt: 是否显示图像
    :param name: 特征名称
    :param m_list: 特征数组
    :return:
    """
    if type(m_list[0]) is str:
        m_list = common_utils.str_list_2_int(m_list)
    m_list.sort()
    # 底边
    Q1 = np.quantile(m_list, 0.25)
    # 上边
    Q3 = np.quantile(m_list, 0.75)
    # 中位数
    Q2 = np.quantile(m_list, 0.5)
    # 四分位距
    IQR = Q3 - Q1
    # 下边界
    min_num = Q1 - 1.5 * IQR
    # 上边界
    max_num = Q3 + 1.5 * IQR
    # 画图
    if show_plt:
        plt.title(name)
        plt.boxplot(m_list, flierprops={"marker": "*"})
        plt.show()
    return min_num, Q1, Q2, Q3, max_num


def get_sequence_divide_mean(m_df: pd.DataFrame, names: list):
    """
    计算每个值和当前列平均值的商
    :param m_df: 数据表
    :param names: 名称
    :return: 算好的表格
    """
    t_df = pd.DataFrame(m_df).copy()
    # 对于每一列计算
    for name in tqdm(names):
        # 计算当前列的平均值
        # t_df_mean = list_to_float_df(t1_df.to_list()).mean()[0]
        s = t_df[name]
        m = s.mean()
        t_df_mean = m
        # 当前列每一个值 / 计算得到的平均值
        for i in range(len(t_df[name])):
            if type(t_df.loc[i, name]) == str:
                t_df.loc[i, name] = float(t_df.loc[i, name].strip())
            t_df.loc[i, name] = t_df.loc[i, name] / t_df_mean
    return t_df


def get_sequence(m_df, names, csv_path):
    """
    得到计算好的序列
    :param m_df: 数据表
    :param names: 名称
    :param csv_path: 路径
    :return: 处理好的序列
    """
    if os.path.exists(csv_path):
        sequence = pd.read_csv(csv_path)
    else:
        sequence = get_sequence_divide_mean(m_df, [names])
        sequence.to_csv(csv_path, index=False)
    return sequence


def get_mean_sequence(p_df: pd.Series, c_df: pd.DataFrame):
    # 得到母序列均值商
    parent = get_sequence(p_df, get_series_name(p_df), parent_mean_sequence)
    # 得到子序列均值商
    child = get_sequence(c_df, c_df.columns.values[1:], child_mean_sequence)
    return parent, child


def cal_ab(p_df: pd.Series, c_df: pd.DataFrame):
    local_min = 100
    local_max = 0
    for name in c_df.columns.values[1:]:
        # 对每一个子序列，计算对应值
        sequence = c_df[name]
        for i in range(len(sequence)):
            # 每一行的数相减
            t = abs(float(c_df.loc[i, name]) - float(p_df[get_series_name(p_df)][i]))
            if t < local_min:
                local_min = t
            if t > local_max:
                local_max = t
    return local_min, local_max


def cal_relevancy(p_df: pd.Series, c_df: pd.DataFrame, a, b, p):
    copy_child = c_df.copy()
    if os.path.exists(grey_relation_sequence):
        copy_child = pd.read_csv(grey_relation_sequence)
    else:
        for name in tqdm(c_df.columns.values[1:]):
            # 对每一个子序列，计算对应值
            sequence = copy_child[name]
            for i in range(len(sequence)):
                # 计算关联度
                copy_child.loc[i, name] = (a + p * b) / (
                        abs(copy_child.loc[i, name] - p_df[get_series_name(p_df)][i]) + p * b)
        copy_child.to_csv(grey_relation_sequence, index=False)
    return copy_child


def get_gray_relation_dataframe(p_df: pd.Series, c_df: pd.DataFrame, p=0.5):
    """
    通过母子序列得到灰色关联度矩阵
    :param p_df: 母序列
    :param c_df: 子序列
    :param p:    关联系数
    :return:
    """
    print('获取母子序列均值商')
    parent_sequence, child_sequence = get_mean_sequence(p_df, c_df)
    print('计算母子矩阵差值矩阵极值')
    a, b = cal_ab(parent_sequence, child_sequence)
    print('计算灰色关联度矩阵')
    relevancy_df = cal_relevancy(parent_sequence, child_sequence, a, b, p)
    return relevancy_df


def get_relevancy_top(m_df: pd.DataFrame, count):
    """
    每一个子序列计算平均值，返回前count个数据
    :param count: 获取前几个数据
    :param m_df:
    :return: 一个list
    """
    obj_list = {}
    for name in tqdm(m_df.columns.values[1:]):
        # 对每一个子序列，计算平均值
        m_mean = m_df[name].mean()
        obj_list[name] = m_mean
    # t_df = pd.DataFrame(obj_list)
    obj_list = sorted(obj_list.items(), key=lambda x: x[1], reverse=True)
    gray_name = []
    for t in obj_list:
        gray_name.append(t[0])
    gray_name = gray_name[:count]
    return gray_name


def get_random_forest_result(dataset):
    x, y = dataset.iloc[:, 1:].values, dataset.iloc[:, 0].values
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    feat_labels = dataset.columns[1:]
    forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(x, y.astype('int'))
    return x, forest
