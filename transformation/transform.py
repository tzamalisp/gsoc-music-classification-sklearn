import pandas as pd
from termcolor import colored
import collections
from transformation.utils_preprocessing import list_descr_handler, descr_remover, descr_enumerator, descr_selector
from transformation.utils_scaling import descr_normalizing, descr_gaussianizing

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
        # self.pre_processing()

        self.df_cat = pd.DataFrame()
        self.df_num = pd.DataFrame()

    def cleaner(self):
        cleaning_columns_list = self.config["excludedDescriptors"]
        cleaning_columns_list = list_descr_handler(cleaning_columns_list)
        print("Cleaner for columns: {}".format(cleaning_columns_list))
        self.df = descr_remover(self.df, cleaning_columns_list)
        print("Shape of the df after the data cleaning: \n{}".format(self.df.shape))

    def post_processing(self):
        print(colored("PROCESS: {}".format(self.process), "cyan"))
        print(self.config["processing"][self.process])
        list_preprocesses = []
        for item in self.config["processing"][self.process]:
            list_preprocesses.append(item["transfo"])
        print(list_preprocesses)

        for item in self.config["processing"][self.process]:
            if item["transfo"] == "remove":
                print(colored("Proccessing"
                              " --> REMOVE", "yellow"))
                # print(item["params"])
                remove_list = list_descr_handler(item["params"]["descriptorNames"])
                print(remove_list)
                self.df = descr_remover(self.df, remove_list)
                print("items removed related to: {}".format(remove_list))

        for item in self.config["processing"][self.process]:
            if item["transfo"] == "enumerate":
                print(colored("Proccessing --> ENUMERATE", "yellow"))
                enumerate_list = list_descr_handler(item["params"]["descriptorNames"])
                print(enumerate_list)
                self.df_num, self.df_cat = descr_enumerator(self.df, enumerate_list,
                                                            exports_path=self.exports_path,
                                                            mode=self.mode)
                print("items enumerated related to: {}".format(enumerate_list))

        for item in self.config["processing"][self.process]:
            if item["transfo"] == "normalize":
                print(colored("Proccessing --> NORMALIZE", "yellow"))
                self.df_num = descr_normalizing(feats_data=self.df_num,
                                                processing=item,
                                                config=self.config,
                                                exports_path=self.exports_path,
                                                train_process=self.process,
                                                mode=self.mode
                                                )

        for item in self.config["processing"][self.process]:
            if item["transfo"] == "gaussianize":
                print(colored("Proccessing --> GAUSSIANIZE", "yellow"))
                self.df_num = descr_gaussianizing(feats_data=self.df_num,
                                                  processing=item,
                                                  config=self.config,
                                                  exports_path=self.exports_path,
                                                  train_process=self.process,
                                                  mode=self.mode
                                                  )

        for item in self.config["processing"][self.process]:
            if item["transfo"] == "select":
                print(colored("Proccessing --> SELECT", "yellow"))
                select_list = list_descr_handler(item["params"]["descriptorNames"])
                print(select_list)
                self.df_num = descr_selector(self.df_num, select_list)
                print(self.df_num)
                print("items selected related to: {}".format(select_list))
                self.df = self.df_num
                print(self.df)

        if "select" not in list_preprocesses:
            self.df = pd.concat([self.df_num, self.df_cat], axis=1)
            print(self.df.head())
        return self.df
