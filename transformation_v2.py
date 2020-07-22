import os
import argparse
from termcolor import colored
from transformation.load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from classification.data_processing import DataProcessing
from classification.classification_task_manager import ClassificationTaskManager
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from transformation.utils_preprocessing import list_descr_handler, descr_remover, descr_handling
from transformation.utils_preprocessing import descr_list_categorical_selector
from transformation.utils_preprocessing import descr_list_remover

from transformation.utils_scaling import descr_scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


def classification_project():
    config_data = load_yaml("configuration_copy.yaml")
    gt_files_list = ListGroundTruthFiles(config_data).list_gt_filenames()
    print(gt_files_list)
    print("LOAD GROUND TRUTH")
    for gt_file in gt_files_list:
        gt_data = GroundTruthLoad(config_data, gt_file)
        df_fg_data = gt_data.export_gt_tracks()
        print(colored("Type of exported GT data exported: {}".format(type(df_fg_data)), "green"))
        class_name = gt_data.export_train_class()

        data_processing_obj = DataProcessing(config=config_data,
                                             dataset=df_fg_data,
                                             class_name=class_name
                                             )
        tracks_shuffled = data_processing_obj.shuffle_tracks_data()
        print(colored("SHUFFLED TRACKS:", "green"))
        print(tracks_shuffled[:4])
        print()

        X, y = data_processing_obj.exporting_classification_data()
        print()
        print()
        print(X.head())
        print(len(X.columns))

        X_transformed = TransformV2(config=config_data, X=X)
        X_transformed.transform_basic()


class TransformV2:
    def __init__(self, config, X):
        self.config = config
        self.X = X

    def transform_basic(self):
        # cleaning
        config_cleaning_columns_list = self.config["excludedDescriptors"]
        config_cleaning_columns_list = list_descr_handler(config_cleaning_columns_list)
        cleaning_attribs = descr_list_remover(self.X, config_cleaning_columns_list)
        print(config_cleaning_columns_list)

        pre_processes = self.config["preprocessing"]["basic"]
        cat_attribs = []
        selecting_attribs = []
        for process in pre_processes:
            print(process)
            if process["transfo"] == "enumerate":
                config_cat_list = process["params"]["descriptorNames"]
                config_cat_list = list_descr_handler(config_cat_list)
                cat_attribs = descr_list_categorical_selector(self.X, config_cat_list)
                print(cat_attribs)
            if process["transfo"] == "remove":
                config_remove_list = process["params"]["descriptorNames"]
                config_remove_list = list_descr_handler(config_remove_list)
                selecting_attribs = descr_list_remover(self.X, config_remove_list)
                print(selecting_attribs)
                print(len(selecting_attribs))

        num_pipeline = Pipeline([
            ('remover', DataFrameSelector(selecting_attribs))
        ])

        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('cat_encoder', OneHotEncoder(sparse=False)),
        ])

        tracks_num_prepared = num_pipeline.fit_transform(self.X)

        print(tracks_num_prepared)

        # print(tracks_cat_prepared)
        #
        # enc = OneHotEncoder(sparse=False)
        # tracks_cat_prepared_1hot = enc.fit_transform(tracks_cat_prepared)
        # print(tracks_cat_prepared_1hot)
        # print(tracks_cat_prepared_1hot.shape)
        # print(enc.categories_)





# Create a class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Create a class to select numerical or categorical columns
class DataFrameRemover(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


if __name__ == '__main__':
    classification_project()