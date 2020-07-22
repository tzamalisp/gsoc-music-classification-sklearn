import glob, os
import pathlib
import json
import yaml

import collections
from pprint import pprint
import dask
from transformation.utils_preprocessing import list_descr_handler, descr_remover, descr_handling
from transformation.utils_scaling import descr_scaling

# avoid the module's method call deprecation
try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


class Transform:
    def __init__(self, config, df, process, exports_path, mode):
        self.config = config
        self.df = df
        self.process = process
        self.exports_path = exports_path
        self.mode = mode
        self.cleaner()
        self.pre_processing()

    def cleaner(self):
        cleaning_columns_list = self.config["excludedDescriptors"]
        cleaning_columns_list = list_descr_handler(cleaning_columns_list)
        print("Cleaner for columns: {}".format(cleaning_columns_list))
        self.df = descr_remover(self.df, cleaning_columns_list)
        print("Shape of the df after the data cleaning: \n{}".format(self.df.shape))

    def pre_processing(self):
        print("Processing: {}".format(self.process))
        print(self.config["processing"][self.process])
        print()
        if "preprocess" in self.config["processing"][self.process].keys():
            print("Preprocessing steps found.")
            preprocess_steps = self.config["processing"][self.process]["preprocess"]
            for step in preprocess_steps:
                self.df = descr_handling(df=self.df, processing=step)
        else:
            print("No preprocessing steps found.")
        print("Shape of the df after the data preprocessing: \n{}".format(self.df.shape))
        return self.df

    def post_processing(self):
        if "postprocess" in self.config["processing"][self.process].keys():
            print("Postprocessing steps found.")
            postprocess_steps = self.config["processing"][self.process]["postprocess"]
            print("Postprocess steps: {}".format(postprocess_steps))
            for step in postprocess_steps:
                print("Scale process: {}".format(step))
                self.df = descr_scaling(feats_data=self.df,
                                        processing=step,
                                        config=self.config,
                                        exports_path=self.exports_path,
                                        train_process=self.process,
                                        mode=self.mode
                                        )
        else:
            print("No postprocessing steps found.")
        print("Shape of the df after the postprocessing (scaling): \n{}".format(self.df.shape))
        return self.df
