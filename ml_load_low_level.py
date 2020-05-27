import json
import pandas as pd
import collections
from ml_load_groung_truth import GroundTruthLoad


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


class FeaturesDf:
    """
    Features DataFrame object by the JSON low-level data.
     Attributes:
            df_tracks (Pandas DataFrame): The tracks DataFrame that contains the track name, track low-level path,
                                        label, etc.
    """
    def __init__(self, df_tracks):
        self.df_tracks = df_tracks
        self.df_feats_tracks = pd.DataFrame()
        self.df_full_tracks = pd.DataFrame()

        self.create_low_level_df()

    def create_low_level_df(self):
        """
        Creates the low-level DataFrame. Cleans also the low-level data from the unnecessary features before creating
        the DF.
        """
        list_feats_tracks = []

        list_feats_tracks.clear()  # clear the list if it not empty

        counter_items_transformed = 0
        for index, row in self.df_tracks.iterrows():
            f = open(row['track_path'])
            data_feats_item = json.load(f)

            # remove unnecessary features data
            if 'metadata' in data_feats_item:
                del data_feats_item['metadata']
            if 'beats_position' in data_feats_item['rhythm']:
                del data_feats_item['rhythm']['beats_position']

            # data dictionary transformed to a fully flattened dictionary
            data_feats_item = flatten_dict_full(data_feats_item)

            # append to a full tracks features pandas df
            list_feats_tracks.append(dict(data_feats_item))

            counter_items_transformed += 1

        print('Items parsed and transformed:', counter_items_transformed)
        # The type of the dictionary's keys list is: <class 'dict_keys'>
        print('Type of the List of features keys:', type(list_feats_tracks[0].keys()))
        # The dictionary's keys list is transformed to type <class 'list'>
        self.df_feats_tracks = pd.DataFrame(list_feats_tracks, columns=list(list_feats_tracks[0].keys()))

    def concatenate_dfs(self):
        """
        :return:
        DataFrame: The tracks with all the ground truth data and the corresponding low-level data flattened.
        """
        print('Tracks DataFrame columns:', self.df_tracks.columns)
        self.df_tracks.drop(labels=['track_path'], axis=1, inplace=True)
        self.df_full_tracks = pd.concat([self.df_tracks, self.df_feats_tracks], axis=1)
        return self.df_full_tracks

    def check_feats_df_info(self):
        """
        Todo: description
        :return:
        """
        print('Features DataFrame head:')
        print(self.df_feats_tracks.head())
        print()
        print('Information:')
        print(self.df_feats_tracks.info())
        print()
        print('Shape:')
        print(self.df_feats_tracks.shape)

    def check_full_df_info(self):
        """
        Todo: description
        :return:
        """
        print('Full tracks DataFrame head:')
        print(self.df_full_tracks.head())
        print()
        print('Information:')
        print(self.df_full_tracks.info())
        print()
        print('Shape:')
        print(self.df_full_tracks.shape)


if __name__ == '__main__':
    gt_data = GroundTruthLoad()
    feat_data = FeaturesDf(gt_data.create_df_tracks())
    feat_data.check_feats_df_info()
