import json
import pandas as pd
import collections
from ml_load_groung_truth import GroundTruthLoad
from utils import DfChecker
from utils import load_yaml
import random


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


def shuffle_data(df_ml_data, config):
    """

    :param df_ml_data: (Pandas DataFrame) the data to be shuffled
    :param config: (dict) the configuration data
    :return: (NumPy array) the shuffled data
    """
    df_ml_cols = df_ml_data.columns
    # convert DataFrame to NumPy array
    ml_values = df_ml_data.values
    # shuffle the data
    random.seed(a=config.get("random_seed"))
    random.shuffle(ml_values)
    # convert the NumPy array to DF
    df_ml_shuffle = pd.DataFrame(data=ml_values, columns=df_ml_cols)
    return df_ml_shuffle


class FeaturesDf:
    """
    Features DataFrame object by the JSON low-level data.
     Attributes:
            df_tracks (Pandas DataFrame): The tracks DataFrame that contains the track name, track low-level path,
                                        label, etc.
    """
    def __init__(self, df_tracks, class_name, config):
        self.df_tracks = df_tracks
        self.class_name = class_name
        self.config = config
        self.list_feats_tracks = []
        self.counter_items_transformed = 0
        self.df_feats_tracks = pd.DataFrame()
        self.df_feats_label = pd.DataFrame()

        self.create_low_level_df()

    def create_low_level_df(self):
        """
        Creates the low-level DataFrame. Cleans also the low-level data from the unnecessary features before creating
        the DF.

        :return:
        DataFrame: low-level features Daa=taFrame from all the tracks in the collection.
        """
        # clear the list if it not empty
        self.list_feats_tracks.clear()
        for index, row in self.df_tracks.iterrows():
            # print(row['track_path'])
            f = open(row['track_path'])
            # data_feats_item = {}
            # if f:
            data_feats_item = json.load(f, strict=False)

            # remove unnecessary features data
            # if 'metadata' in data_feats_item:
            #     del data_feats_item['metadata']
            if 'beats_position' in data_feats_item['rhythm']:
                del data_feats_item['rhythm']['beats_position']

            # data dictionary transformed to a fully flattened dictionary
            data_feats_item = flatten_dict_full(data_feats_item)

            # append to a full tracks features pandas df
            self.list_feats_tracks.append(dict(data_feats_item))

            self.counter_items_transformed += 1

        # The dictionary's keys list is transformed to type <class 'list'>
        self.df_feats_tracks = pd.DataFrame(self.list_feats_tracks, columns=list(self.list_feats_tracks[0].keys()))
        print("COLUMNS CONTAIN OBJECTS", self.df_feats_tracks.select_dtypes(include=['object']).columns)
        return self.df_feats_tracks

    def check_processing_info(self):
        """
        Prints some information about the low-level data to DataFrame transformation step and its middle processes.
        :return:
        """
        print('Items parsed and transformed:', self.counter_items_transformed)
        # The type of the dictionary's keys list is: <class 'dict_keys'>
        print('Type of the list of features keys:', type(self.list_feats_tracks[0].keys()))
        # The dictionary's keys list is transformed to type <class 'list'>
        print('Confirm the type of list transformation of features keys', type(list(self.list_feats_tracks[0].keys())))

    def concatenate_dfs(self):
        """
        :return:
        DataFrame: The tracks with all the ground truth data and the corresponding low-level data flattened.
        """
        self.df_tracks.drop(labels=['track_path', 'json_directory', 'track'], axis=1, inplace=True)
        print()
        print()
        print("CONCATENATING")
        print("TRACKS SHAPE:", self.df_tracks.shape)
        print("LOW LEVEL:", self.df_feats_tracks.shape)

        self.df_feats_label = pd.concat([self.df_tracks, self.df_feats_tracks], axis=1)
        print("FULL:", self.df_feats_label.shape)
        print("COLUMNS CONTAIN OBJECTS", self.df_feats_label.select_dtypes(include=['object']).columns)
        # # gaia imitation shuffling (in case we want to shuffle the data exactly as the gaia tool does)
        # if self.config["gaia_imitation"] is True:
        #     print("DF CONCATENATION BEFORE SHUFFLING")
        #     print("head:")
        #     print(self.df_feats_label.iloc[:, 0].head(10))
        #     print("tail:")
        #     print(self.df_feats_label.iloc[:, 0].tail(10))
        #     self.df_feats_label = shuffle_data(df_ml_data=self.df_feats_label, config=self.config)
        #     print("DF CONCATENATION AFTER SHUFFLING")
        #     print("head:")
        #     print(self.df_feats_label.iloc[:, 0].head(10))
        #     print("tail:")
        #     print(self.df_feats_label.iloc[:, 0].tail(10))
        return self.df_feats_label


if __name__ == '__main__':
    print("LOW LEVEL DF LOAD SCRIPT")
    #
    config_data = load_yaml()
    df_gt_data = GroundTruthLoad(config_data, "groundtruth.yaml").create_df_tracks()
    print(df_gt_data.shape)
    df_feat_data = FeaturesDf(df_tracks=df_gt_data, config=config_data).concatenate_dfs()
