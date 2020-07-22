import os
import re
import pandas as pd
import collections
from sklearn.preprocessing import OneHotEncoder
import joblib
from utils import load_yaml, FindCreateDirectory, TrainingProcesses

def flatten_dict_full(dictionary, sep="_"):
    """

    :param dictionary:
    :param sep:
    :return:
    """
    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):
        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(dictionary)

    return obj


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
    print(columns_sel_list)
    print(len(columns_sel_list))
    df_select_descr = df[columns_sel_list]
    return df_select_descr


def descr_enumerator(df, descr_enumerate_list, exports_path, mode):
    """

    :param df:
    :param descr_enumerate_list:
    :return:
    """
    models_path = FindCreateDirectory(os.path.join(exports_path, "models")).inspect_directory()
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
    # print("ENUMERATED FEATS:")

    # if mode == "train":
    #     encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    #     encoder.fit(df_cat)
    #     transformed = encoder.transform(df_cat)
    #     joblib.dump(encoder, os.path.join(models_path, "encoder.pkl"))
    #     print("Categories enumerated: {}".format(encoder.categories_))
    #     print("Shape of enumerated numpy array: {}".format(transformed.shape))
    #     df_cat_oh = pd.DataFrame(data=transformed)
    #     print(df_cat_oh.head())
    # elif mode == "predict":
    #     encoder = joblib.load(os.path.join(models_path, "encoder.pkl"))
    #     print("OneHotEncoder loaded..")
    #     transformed = encoder.transform(df_cat)
    #     df_cat_oh = pd.DataFrame(data=transformed)
    #     print(df_cat_oh.head())

    df.drop(labels=columns_enum_list, axis=1, inplace=True)
    # df_num_oh = pd.concat([df, df_cat_oh], axis=1)
    return df, df_cat_oh


def descr_handling(df, processing, exports_path, mode):
    """

    :param df:
    :param processing:
    :param exports_path:
    :param mode:
    :return:
    """
    if processing["transfo"] == "remove":
        remove_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_remover(df, remove_list)
        print("items removed related to: {}".format(remove_list))
        print()
    if processing["transfo"] == "enumerate":
        enumerate_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_enumerator(df, enumerate_list, exports_path=exports_path, mode=mode)
        print("items enumerated related to: {}".format(enumerate_list))
        print()
    if processing["transfo"] == "select":
        select_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_selector(df, select_list)
        print("items selected related to: {}".format(select_list))
        print()
    return df


def descr_list_categorical_selector(df, descr_enumerate_list):
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
    return columns_enum_list
    # df_cat = df[columns_enum_list]
    # return df_cat


def descr_list_remover(df, descr_remove_list, ):
    """

    :param df:
    :param descr_remove_list:
    :return:
    """
    columns_list = list(df.columns)
    columns_del_list = []
    for item in descr_remove_list:
        for del_item in columns_list:
            if not re.search(item, del_item):
                columns_del_list.append(del_item)
    return columns_del_list
