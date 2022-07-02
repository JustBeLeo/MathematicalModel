from _csv import reader
import pandas as pd


def get_objs_in_csv(path):
    with open(path) as r:
        readers = reader(r, delimiter=",")
        lines = list(readers)
        # 获取第一行所有的名称
        names = get_column_name(lines)
        # 去除第一行
        lines = lines[1:]
        m_list = get_obj_list(lines, names)
        return names, pd.DataFrame(m_list)


def get_obj_list(lines, names):
    """
    获取对象数组
    :param lines: 每行的数据
    :param names: 每列的名称
    :return:
    """
    m_list = []
    for line in lines:
        a = {}
        for i in range(len(names)):
            if i != 0:
                a[names[i]] = float(line[i])
            else:
                a[names[i]] = line[i]
        m_list.append(a)
    return m_list


def get_column_name(lines):
    names = lines[0]
    new_names = []
    for name in names:
        if not name.isspace():
            new_names.append(name)
    names = new_names
    return names
