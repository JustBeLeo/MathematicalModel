import numpy as np
import pandas as pd
from tqdm import tqdm


def str_list_2_int(m_list: list):
    t_list = []
    for x in m_list:
        t_list.append(float(x))
    return t_list


def list_to_float_df(m_list: list):
    """
    将list转换为float df
    :param m_list:
    :return:
    """
    t_list = []
    for x in m_list:
        t_list.append(float(x))
    return pd.DataFrame(t_list)


def convert_float_df(m_df: pd.DataFrame):
    names = m_df.columns.values
    for name in names:
        for i in range(len(m_df[name])):
            m_df[name][i] = float(m_df[name][i])
    return m_df


def count_zero(m_list: list):
    """
    计算当前数组0的占比
    :param m_list:
    :return:
    """
    if type(m_list[0]) is str:
        m_list = str_list_2_int(m_list)
    zero_count = 0
    for x in m_list:
        if x == 0:
            zero_count += 1
    return zero_count / len(m_list)


def remove_names(m_df: pd.DataFrame, origin_names: list, names: list):
    t_df = m_df.copy()
    for name in names:
        t_df.drop(name, axis=1, inplace=True)
        origin_names.remove(name)
    return t_df


def split_by_ratio(m_list, *ratios):
    """
    按比例切分列表
    :param m_list: 列表
    :param ratios: 比例，如0.5,0.5
    :return:
    """
    m_list = np.random.permutation(m_list)
    ind = np.add.accumulate(np.array(ratios) * len(m_list)).astype(int)
    return [x.tolist() for x in np.split(m_list, ind)][:len(ratios)]
