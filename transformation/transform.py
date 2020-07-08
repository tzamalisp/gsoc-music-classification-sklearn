import glob, os
import pathlib
import json
import yaml

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from pprint import pprint
import re
import dask

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

# avoid the module's method call deprecation
try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


def list_descr_handler(descr_list):
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
    columns_list = list(df.columns)
    columns_del_list = []
    for item in descr_remove_list:
        for del_item in columns_list:
            if re.search(item, del_item):
                columns_del_list.append(del_item)
    df_used_descr = df.drop(columns=columns_del_list, axis=1)
    return df_used_descr


def descr_enumerator(df, descr_enumerate_list):
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
    columns_list = list(df.columns)
    columns_sel_list = []
    for item in descr_select_list:
        for sel_item in columns_list:
            if re.search(item, sel_item):
                columns_sel_list.append(sel_item)
    df_select_descr = df[columns_sel_list]
    return df_select_descr


def descr_handling(df, processing):
    if processing["transfo"] == "remove":
        remove_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_remover(df, remove_list)
        print("items removed related to: {}".format(remove_list))
        print()
    if processing["transfo"] == "enumerate":
        enumerate_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_enumerator(df, enumerate_list)
        print("items enumerated related to:: {}".format(enumerate_list))
        print()
    if processing["transfo"] == "select":
        select_list = list_descr_handler(processing["params"]["descriptorNames"])
        df = descr_selector(df, select_list)
        print("items selected related to: {}".format(select_list))
        print()
    return df


def descr_scaling(feats_data, processing):
    # Normalize dataset
    if processing["transfo"] == "normalize":
        feats_data_columns = feats_data.columns
        print("length of df features columns: {}".format(len(feats_data_columns)))
        # normalize
        normalizer = MinMaxScaler()
        normalizer.fit(feats_data)
        feats_data_normalized = normalizer.transform(feats_data)
        print("Type of normalized data: {}".format(type(feats_data_normalized)))
        feats_data = pd.DataFrame(data=feats_data_normalized, columns=feats_data_columns)
        print("Type of normalized data after conversion: {}".format(type(feats_data)))
        print(feats_data.iloc[:, 0].head())
        sns.distplot(feats_data.iloc[:, 0])
        print("Normalization process completed.")
        print()
        print()

    # Gaussianize dataset
    if processing["transfo"] == "gaussianize":
        feats_data_columns = feats_data.columns
        select_list = list_descr_handler(processing["params"]["descriptorNames"])
        print("Selection list: {}".format(select_list))
        print("Input DF - no. of columns: {}".format(len(feats_data.columns)))
        df_gauss = descr_selector(df=feats_data, descr_select_list=select_list)
        df_gauss_columns = df_gauss.columns
        print("Gaussian DF - no. of columns: {}".format(len(df_gauss_columns)))
        df_no_gauss = feats_data.drop(df_gauss_columns, axis=1)
        print("No Gaussian DF - no. of columns: {}".format(len(df_no_gauss.columns)))
        # gaussianize
        gaussianizer = QuantileTransformer(n_quantiles=1000)
        gaussianizer.fit(df_gauss)
        feats_data_gaussianized = gaussianizer.transform(df_gauss)
        print("Type of gaussianized data: {}".format(type(feats_data_gaussianized)))
        feats_data_gaussianized = pd.DataFrame(data=feats_data_gaussianized, columns=df_gauss_columns)
        feats_data = pd.concat([feats_data_gaussianized, df_no_gauss], axis=1)
        print("Output DF - no. of columns: {}".format(len(feats_data.columns)))
        print(feats_data.iloc[:, 0].head())
        sns.distplot(feats_data.iloc[:, 0])
        print("Gaussianization process completed.")
    return feats_data


class Transform:
    def __init__(self, config, df, process):
        self.config = config
        self.df = df
        self.process = process

    def cleaner(self):
        cleaning_columns_list = self.config["excludedDescriptors"]
        cleaning_columns_list = list_descr_handler(cleaning_columns_list)
        print("cleaner for columns: {}".format(cleaning_columns_list))
        self.df = descr_remover(self.df, cleaning_columns_list)

    def pre_processing(self):
        print("Processing: {}".format(self.process))
        print(self.config["processing"][self.process])
        print()
        if "preprocess" in self.config["processing"][self.process].keys():
            print("Preprocessing steps found. Time to preprocess the DF and return.")
            print()
            preprocess_steps = self.config["processing"][self.process]["preprocess"]
            for step in preprocess_steps:
                self.df = descr_handling(self.df, step)
        return self.df

    def post_processing(self):
        if "postprocess" in self.config["processing"][self.process].keys():
            print("Postprocessing steps found.")
            postprocess_steps = self.config["processing"][self.process]["postprocess"]
            print("Postprocess steps: {}".format(postprocess_steps))
            for step in postprocess_steps:
                self.df = descr_scaling(self.df, step)
        return self.df
