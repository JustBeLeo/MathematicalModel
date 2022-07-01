import os

from d_2021.config.path import *
from d_2021.utils.import_utils import *
from d_2021.utils.math_utils import *
from d_2021.utils.common_utils import *
from tqdm import *


def data_pre_threat():
    if os.path.exists(md_df_path):
        md_df_3 = pd.read_csv(md_df_path)
        md_names = md_df_3.columns.tolist()
    else:
        md_names, md_df = get_objs_in_csv("../dataset/train/Molecular_Descriptor.csv")
        print('总特征数:', len(md_names) - 1)
        error_names = []
        for md_name in md_names[1:]:
            row = md_df[md_name]
            row = str_list_2_int(row)
            min_num, Q1, Q2, Q3, max_num = get_box_plot(row, md_name, False)
            # 计算异常值数量
            error_count = 0
            for n in row:
                if n < min_num or n > max_num:
                    error_count += 1
            if error_count > 100:
                error_names.append(md_name)
        print('异常值数量:', len(error_names))
        md_df_2 = remove_names(md_df, md_names, error_names)
        # 含0大于0.9的列
        zero_names = []
        for md_name in md_names[1:]:
            row = md_df_2[md_name]
            p = count_zero(row)
            if p > 0.9:
                zero_names.append(md_name)
        print('0值大于90%数量:', len(zero_names))
        md_df_3 = remove_names(md_df, md_names, zero_names)
        print('筛选后总数', len(md_names) - 1)
        md_df_3.to_csv(md_df_path, index=False)
    return md_df_3, md_names


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
        sequence = get_sequence_divide_mean(m_df, names)
        sequence.to_csv(csv_path, index=False)
    return sequence


def get_mean_sequence():
    era_names, era_df = get_objs_in_csv("../dataset/train/ERa_activity.csv")
    # 得到母序列均值商
    parent = get_sequence(era_df, ['pIC50'], csv1_path)
    # 得到子序列均值商
    child_names1 = pre_threat_names[1:]
    child = get_sequence(pre_threat_df, child_names1, csv2_path)
    return parent, child, child_names1


def cal_ab():
    local_min = 100
    local_max = 0
    for name in child_names:
        # 对每一个子序列，计算对应值
        sequence = child_sequence[name]
        for i in range(len(sequence)):
            # 每一行的数相减
            t = abs(sequence[i] - parent_sequence['pIC50'][i])
            if t < local_min:
                local_min = t
            if t > local_max:
                local_max = t
    return local_min, local_max


def cal_relevancy():
    copy_child = child_sequence.copy()
    p = 0.5
    for name in tqdm(child_names):
        # 对每一个子序列，计算对应值
        sequence = copy_child[name]
        for i in range(len(sequence)):
            # 计算关联度
            sequence[i] = (a + p * b) / (abs(sequence[i] - parent_sequence['pIC50'][i]) + p * b)
    copy_child.to_csv(csv3_path, index=False)
    return copy_child


# 预处理后的Molecular_Descriptor
# 一、 数据清洗 1. 去除异常值 2. 去除0多的值
pre_threat_df, pre_threat_names = data_pre_threat()
# 二、 对变量进行预处理
parent_sequence, child_sequence, child_names = get_mean_sequence()
# 三、计算灰色关联度
# 计算极大极小值
a, b = cal_ab()
relevancy_df = cal_relevancy()
