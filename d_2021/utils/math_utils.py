import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from d_2021.utils import common_utils
from d_2021.utils.common_utils import to_float_df


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
    list_size = len(m_list)
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
    t_df = m_df.copy()
    # 对于每一列计算
    for name in tqdm(names):
        t1_df = t_df[name]
        # 计算当前列的平均值
        t_df_mean = to_float_df(t1_df.to_list()).mean()[0]
        # 当前列每一个值 / 计算得到的平均值
        for i in range(len(t1_df)):
            if type(t1_df[i]) == str:
                t1_df[i] = float(t1_df[i].strip())
            t1_df[i] = t1_df[i] / t_df_mean
    return t_df[names]
