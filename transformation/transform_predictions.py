import pandas as pd
from termcolor import colored
import collections
import joblib
import os

from transformation.utils_preprocessing import list_descr_handler, descr_enumerator, descr_selector
from transformation.utils_scaling import descr_normalizing, descr_gaussianizing
from transformation.utils_preprocessing import cleaner, descr_remover, feats_selector_list
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

# avoid the module's method call deprecation
try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


class TransformPredictions:
    def __init__(self, config, df_feats, process, exports_path):
        self.config = config
        self.df_feats = df_feats
        self.process = process
        self.exports_path = exports_path

        self.list_features = []
        self.feats_cat_list = []
        self.feats_num_list = []

        self.feats_prepared = []

    def post_processing(self):
        print(colored("PROCESS: {}".format(self.process), "cyan"))
        # list_preprocesses = []

        print(self.df_feats)
        print("Shape of DF:", self.df_feats.shape)

        self.list_features = list(self.df_feats.columns)

        exports_dir = os.path.join(self.exports_path, "models")

        # clean list
        print(colored("Cleaning..", "yellow"))
        cleaning_conf_list = list_descr_handler(self.config["excludedDescriptors"])
        print("cleaning list:", cleaning_conf_list)
        feats_clean_list = feats_selector_list(self.df_feats.columns, cleaning_conf_list)
        self.list_features = [x for x in self.df_feats.columns if x not in feats_clean_list]
        print("List after cleaning some feats: {}".format(len(self.list_features), "blue"))

        # remove list
        print(colored("Removing unnecessary features..", "yellow"))
        if self.config["processing"][self.process][0]["transfo"] == "remove":
            remove_list = list_descr_handler(self.config["processing"][self.process][0]["params"]["descriptorNames"])
            feats_remove_list = feats_selector_list(self.df_feats.columns, remove_list)
            self.list_features = [x for x in self.list_features if x not in feats_remove_list]
            print("List after removing unnecessary feats: {}".format(len(self.list_features), "blue"))

        # enumerate list
        print(colored("Removing unnecessary features..", "yellow"))
        if self.config["processing"][self.process][1]["transfo"] == "enumerate":
            enumerate_list = list_descr_handler(self.config["processing"][self.process][1]["params"]["descriptorNames"])
            self.feats_cat_list = feats_selector_list(self.list_features, enumerate_list)
            print("Enumerating feats: {}".format(self.feats_cat_list))
            self.feats_num_list = [x for x in self.list_features if x not in self.feats_cat_list]
            print("List Num feats: {}".format(len(self.feats_num_list)))
            print("List Cat feats: {}".format(len(self.feats_cat_list), "blue"))

        # BASIC
        if self.process == "basic":
            print(colored("Process doing: {}".format(self.process), "green"))
            print("List post-Num feats: {}".format(len(self.feats_num_list)))

            # load pipeline
            full_pipeline = joblib.load(os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

            self.feats_prepared = full_pipeline.transform(self.df_feats)

        # LOW-LEVEL or MFCC
        if self.process == "lowlevel" or self.process == "mfcc":
            print(colored("Process doing: {}".format(self.process), "green"))
            sel_list = list_descr_handler(self.config["processing"][self.process][2]["params"]["descriptorNames"])
            self.feats_num_list = feats_selector_list(self.feats_num_list, sel_list)
            print("List post-Num feats: {}".format(len(self.feats_num_list)))

            # load pipeline
            full_pipeline = joblib.load(os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

            self.feats_prepared = full_pipeline.transform(self.df_feats)

        # NOBANDS
        if self.process == "nobands":
            print(colored("Process doing: {}".format(self.process), "green"))
            sel_list = list_descr_handler(self.config["processing"][self.process][2]["params"]["descriptorNames"])
            feats_rem_list = feats_selector_list(self.df_feats, sel_list)
            self.feats_num_list = [x for x in self.feats_num_list if x not in feats_rem_list]
            print("List post-Num feats: {}".format(len(self.feats_num_list)))

            # load pipeline
            full_pipeline = joblib.load(os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

            self.feats_prepared = full_pipeline.transform(self.df_feats)

        # NORMALIZED
        if self.process == "normalized":
            print(colored("Process doing: {}".format(self.process), "green"))
            print("List post-Num feats: {}".format(len(self.feats_num_list)))

            # load pipeline
            full_pipeline = joblib.load(os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

            self.feats_prepared = full_pipeline.transform(self.df_feats)

        # GAUSSIANIZED
        if self.process == "gaussianized":
            print(colored("Process doing: {}".format(self.process), "green"))
            gauss_list = list_descr_handler(self.config["processing"][self.process][3]["params"]["descriptorNames"])
            feats_num_gauss_list = feats_selector_list(self.feats_num_list, gauss_list)
            feats_num_no_gauss_list = [x for x in self.feats_num_list if x not in feats_num_gauss_list]

            print("List post-Num feats: {}".format(len(self.feats_num_list)))
            print("List post-Num-Gauss feats: {}".format(len(feats_num_gauss_list)))

            # load pipeline
            full_pipeline = joblib.load(os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

            self.feats_prepared = full_pipeline.transform(self.df_feats)

        return self.feats_prepared


# Create a class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values