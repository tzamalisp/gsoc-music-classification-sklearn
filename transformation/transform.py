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
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# avoid the module's method call deprecation
try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


class Transform:
    def __init__(self, config, df_feats, process, exports_path):
        self.config = config
        self.df_feats = df_feats
        self.process = process
        self.exports_path = exports_path

        self.list_features = []
        self.feats_cat_list = []
        self.feats_num_list = []
        self.df_cat = pd.DataFrame()
        self.df_num = pd.DataFrame()

        self.feats_prepared = []

    def post_processing(self):
        print(colored("PROCESS: {}".format(self.process), "cyan"))
        print(self.config["processing"][self.process])
        # list_preprocesses = []

        self.list_features = list(self.df_feats.columns)

        exports_dir = os.path.join(self.exports_path, "models")

        # clean list
        print(colored("Cleaning..", "yellow"))
        cleaning_conf_list = list_descr_handler(self.config["excludedDescriptors"])
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
            print("List post-Num feats: {}".format(len(self.feats_num_list)))

            num_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_num_list))
            ])

            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_cat_list)),
                ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])

            full_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", num_pipeline),
                ("cat_pipeline", cat_pipeline)
            ])

            self.feats_prepared = full_pipeline.fit_transform(self.df_feats)

            # save pipeline
            joblib.dump(full_pipeline, os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

        # LOW-LEVEL or MFCC
        if self.process == "lowlevel" or self.process == "mfcc":
            sel_list = list_descr_handler(self.config["processing"][self.process][2]["params"]["descriptorNames"])
            self.feats_num_list = feats_selector_list(self.feats_num_list, sel_list)
            print("List post-Num feats: {}".format(len(self.feats_num_list)))

            num_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_num_list))
            ])

            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_cat_list)),
                ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])

            full_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", num_pipeline),
                ("cat_pipeline", cat_pipeline)
            ])

            self.feats_prepared = full_pipeline.fit_transform(self.df_feats)

            # save pipeline
            joblib.dump(full_pipeline, os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

        # NOBANDS
        if self.process == "nobands":
            sel_list = list_descr_handler(self.config["processing"][self.process][2]["params"]["descriptorNames"])
            feats_rem_list = feats_selector_list(self.df_feats, sel_list)
            self.feats_num_list = [x for x in self.feats_num_list if x not in feats_rem_list]
            print("List post-Num feats: {}".format(len(self.feats_num_list)))

            num_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_num_list))
            ])

            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_cat_list)),
                ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])

            full_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", num_pipeline),
                ("cat_pipeline", cat_pipeline)
            ])

            self.feats_prepared = full_pipeline.fit_transform(self.df_feats)

            # save pipeline
            joblib.dump(full_pipeline, os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))

        # NORMALIZED
        if self.process == "normalized":
            print("List post-Num feats: {}".format(len(self.feats_num_list)))
            num_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_num_list)),
                ('minmax_scaler', MinMaxScaler()),
            ])

            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_cat_list)),
                ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])

            full_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", num_pipeline),
                ("cat_pipeline", cat_pipeline)
            ])

            self.feats_prepared = full_pipeline.fit_transform(self.df_feats)

            # save pipeline
            joblib.dump(full_pipeline, os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))


        # GAUSSIANIZED
        if self.process == "gaussianized":
            gauss_list = list_descr_handler(self.config["processing"][self.process][3]["params"]["descriptorNames"])
            feats_num_gauss_list = feats_selector_list(self.feats_num_list, gauss_list)
            feats_num_no_gauss_list = [x for x in self.feats_num_list if x not in feats_num_gauss_list]

            print("List post-Num feats: {}".format(len(self.feats_num_list)))
            print("List post-Num-Gauss feats: {}".format(len(feats_num_gauss_list)))
            print("List post-Num-No-Gauss feats: {}".format(len(feats_num_no_gauss_list)))

            # t = [('minmax', MinMaxScaler(), self.feats_num_list),
            #      ('gauss', QuantileTransformer(n_quantiles=1000), feats_num_gauss_list)]
            # col_transform = ColumnTransformer(transformers=t)
            # print(colored("normalize..", "cyan"))
            num_norm_pipeline = Pipeline([
                ("selector_num", DataFrameSelector(self.feats_num_list)),
                ("minmax_scaler", MinMaxScaler())
            ])

            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(self.feats_cat_list)),
                ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])

            full_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", num_norm_pipeline),
                ("cat_pipeline", cat_pipeline)
            ])

            self.feats_prepared = full_pipeline.fit_transform(self.df_feats)
            print("Feats prepared normalized shape: {}".format(self.feats_prepared.shape))
            # save pipeline
            joblib.dump(full_pipeline, os.path.join(exports_dir, "full_pipeline_{}.pkl".format(self.process)))
            self.df_feats = pd.DataFrame(data=self.feats_prepared)
            columns = list(self.df_feats.columns)
            # print(columns)
            select_rename_list = columns[:len(self.feats_num_list)]
            select_rename_list = self.feats_num_list
            select_no_rename_list = columns[len(self.feats_num_list):]
            print(select_no_rename_list)
            new_feats_columns = select_rename_list + select_no_rename_list
            self.df_feats.columns = new_feats_columns
            print("Normalized Features DF:")
            print(self.df_feats)
            print("Shape: {}".format(self.df_feats.shape))


            num_gauss_pipeline = Pipeline([
                ("gauss_sel_num", DataFrameSelector(feats_num_gauss_list)),
                ("gauss_scaler", QuantileTransformer(n_quantiles=1000))
            ])

            self.feats_prepared = num_gauss_pipeline.fit_transform(self.df_feats)

            # save pipeline
            joblib.dump(num_gauss_pipeline, os.path.join(exports_dir, "gauss_pipeline_{}.pkl".format(self.process)))

        # # remove unnecessary features
        # if self.config["processing"][self.process] == "basic":
        #
        #
        #
        # if self.process == ""

        # for item in self.config["processing"][self.process]:
        #     if item["transfo"] == "remove":
        #         print(colored("Proccessing --> REMOVE", "yellow"))
        #         # print(item["params"])
        #         remove_list = list_descr_handler(item["params"]["descriptorNames"])
        #         print(remove_list)
        #         self.df = descr_remover(self.df, remove_list)
        #         print("items removed related to: {}".format(remove_list))
        #
        # for item in self.config["processing"][self.process]:
        #     if item["transfo"] == "enumerate":
        #         print(colored("Proccessing --> ENUMERATE", "yellow"))
        #         enumerate_list = list_descr_handler(item["params"]["descriptorNames"])
        #         print(enumerate_list)
        #         self.df_num, self.df_cat = descr_enumerator(self.df, enumerate_list,
        #                                                     exports_path=self.exports_path,
        #                                                     mode=self.mode)
        #         print("items enumerated related to: {}".format(enumerate_list))
        #
        # for item in self.config["processing"][self.process]:
        #     if item["transfo"] == "normalize":
        #         print(colored("Proccessing --> NORMALIZE", "yellow"))
        #         self.df_num = descr_normalizing(feats_data=self.df_num,
        #                                         processing=item,
        #                                         config=self.config,
        #                                         exports_path=self.exports_path,
        #                                         train_process=self.process,
        #                                         mode=self.mode
        #                                         )
        #
        # for item in self.config["processing"][self.process]:
        #     if item["transfo"] == "gaussianize":
        #         print(colored("Proccessing --> GAUSSIANIZE", "yellow"))
        #         self.df_num = descr_gaussianizing(feats_data=self.df_num,
        #                                           processing=item,
        #                                           config=self.config,
        #                                           exports_path=self.exports_path,
        #                                           train_process=self.process,
        #                                           mode=self.mode
        #                                           )
        #
        # for item in self.config["processing"][self.process]:
        #     if item["transfo"] == "select":
        #         print(colored("Proccessing --> SELECT", "yellow"))
        #         select_list = list_descr_handler(item["params"]["descriptorNames"])
        #         print(select_list)
        #         self.df_num = descr_selector(self.df_num, select_list)
        #         print(self.df_num)
        #         print("items selected related to: {}".format(select_list))
        #         self.df = self.df_num
        #         print(self.df)
        #
        # if "select" not in list_preprocesses:
        #     self.df = pd.concat([self.df_num, self.df_cat], axis=1)
        #     print(self.df.head())

        return self.feats_prepared


# Create a class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values