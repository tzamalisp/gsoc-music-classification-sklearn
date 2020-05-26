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


path_app = os.path.abspath(os.getcwd())
print('Current path:', path_app)
print('Type:', type(path_app))


def load_local_ground_truth(class_to_search):
    """
    Function to load the the ground truth file
    - The directory with the data should be located inside the app folder location
    """
    with open(os.path.join(path_app, 'acousticbrainz-datasets/{}/metadata/groundtruth.yaml'.format(class_to_search)), 'r') as stream:
        try:
            ground_truth_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # pprint(ground_truth_data)
    print()
    print('Ground truth data class/target:', ground_truth_data['className'])
    print('Ground truth data keys - tracks:', len(ground_truth_data['groundTruth'].keys()))


def count_all_json_files():
    counter = 0
    for root, dirs, files in os.walk(path_app):
        for file in files:
            if file.endswith(".json"):
                # print(os.path.join(root, file))
                counter += 1

    print('counted json files:', counter)


if __name__ == '__main__':
    load_local_ground_truth(class_to_search='danceability')

