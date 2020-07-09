import re
import pandas as pd


def list_descr_handler(descr_list):
    """

    :param descr_list:
    :return:
    """
    keys_list_handle = []
    for item in descr_list:
        if item.endswith(".*"):
            item = item.replace(".*", "_")
        elif item.startswith("*."):
            item = item.replace("*.", "_")
        else:
            item = item.replace("*", "")
        item = item.replace(".", "_")
        keys_list_handle.append(item)
    return keys_list_handle


def descr_remover(df, descr_remove_list):
    """

    :param df:
    :param descr_remove_list:
    :return:
    """
    columns_list = list(df.columns)
    columns_del_list = []
    for item in descr_remove_list:
        for del_item in columns_list:
            if re.search(item, del_item):
                columns_del_list.append(del_item)
    df_used_descr = df.drop(columns=columns_del_list, axis=1)
    return df_used_descr


def descr_enumerator(df, descr_enumerate_list):
    """

    :param df:
    :param descr_enumerate_list:
    :return:
    """
    columns_list = list(df.columns)
    columns_enum_list = []
    for item in descr_enumerate_list:
        for sel_item in columns_list:
            if re.search(item, sel_item):
                columns_enum_list.append(sel_item)
    df_cat = df[columns_enum_list]
    print("No. of columns to enumerate: {}".format(len(df_cat.columns)))
    df_cat_oh = pd.get_dummies(df_cat)
    print("No. of columns after enumeration: {}".format(len(df_cat_oh.columns)))
    print("Columns enumerated: {}".format(df_cat_oh.columns))
    df.drop(labels=columns_enum_list, axis=1, inplace=True)
    df_num_oh = pd.concat([df, df_cat_oh], axis=1)
    return df_num_oh


def descr_selector(df, descr_select_list):
    """

    :param df:
    :param descr_select_list:
    :return:
    """
    columns_list = list(df.columns)
    columns_sel_list = []
    for item in descr_select_list:
        for sel_item in columns_list:
            if re.search(item, sel_item):
                columns_sel_list.append(sel_item)
    df_select_descr = df[columns_sel_list]
    return df_select_descr


def descr_handling(df, processing):
    """

    :param df:
    :param processing:
    :return:
    """
    if processing["transfo"] == "remove":
        remove_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_remover(df, remove_list)
        print("items removed related to: {}".format(remove_list))
        print()
    if processing["transfo"] == "enumerate":
        enumerate_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_enumerator(df, enumerate_list)
        print("items enumerated related to: {}".format(enumerate_list))
        print()
    if processing["transfo"] == "select":
        select_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_selector(df, select_list)
        print("items selected related to: {}".format(select_list))
        print()
    return df
