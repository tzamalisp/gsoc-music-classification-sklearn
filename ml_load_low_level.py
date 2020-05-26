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

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# saving the ML model to pickle file and load it
import pickle

from ml_load_groung_truth import GroundTruthLoad

def flatten_dict_full(d, sep="_"):
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
    recurse(d)

    return obj


class FeaturesDf:
    def __init__(self, df_tracks):
        self.df_tracks = df_tracks

    def create_low_level_df(self):
        list_feats_tracks = []

        list_feats_tracks.clear()  # clear the list if it not empty

        counter_items_transformed = 0
        for index, row in self.df_tracks.iterrows():
            f = open(row['track_path'])
            data_feats_item = json.load(f)

            # remove unnecessary data
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

        df_feats_tracks = pd.DataFrame(list_feats_tracks, columns=list_feats_tracks[0].keys())
        print(df_feats_tracks.head())


if __name__ == '__main__':
    gt_data = GroundTruthLoad(class_to_search='danceability')
    print(FeaturesDf(gt_data.create_df_tracks()))