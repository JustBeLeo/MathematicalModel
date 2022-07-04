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
        md_names, md_df = get_objs_in_csv("../../dataset/train/Molecular_Descriptor.csv")
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


# 预处理后的Molecular_Descriptor
# 一、 数据清洗 1. 去除异常值 2. 去除0多的值
pre_threat_df, pre_threat_names = data_pre_threat()
# 二、 对变量进行预处理
era_names, era_df = get_objs_in_csv("../../dataset/train/ERa_activity.csv")
p_df = era_df['pIC50']
grey_df = get_gray_relation_dataframe(p_df, pre_threat_df)
