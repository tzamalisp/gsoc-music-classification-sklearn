import os
import joblib
from utils import load_yaml, FindCreateDirectory, TrainingProcesses

from transformation.load_low_level import FeaturesDf
from transformation.utils_preprocessing import flatten_dict_full


class Predict:
    def __init__(self, config, track_low_level, process):
        self.config = config
        self.track_low_level = track_low_level
        self.process = process

    def flat_dict(self):
        try:
            if 'beats_position' in self.track_low_level['rhythm']:
                del self.track_low_level['rhythm']['beats_position']
        except Exception as e:
            print("There is no 'rhythm' key in the low level data. Exception:", e)

        # data dictionary transformed to a fully flattened dictionary
        data_feats_item = flatten_dict_full(self.track_low_level)


if __name__ == '__main__':
    import requests
    from pprint import pprint
    config_data = load_yaml("configuration.yaml")
    # pprint(config_data)
    response = requests.get('https://acousticbrainz.org/api/v1/78281677-8ba1-41df-b0f7-df6b024caf13/low-level')
    track = response.json()
    # pprint(track)

    prediction = Predict(config=config_data, track_low_level=track, process="basic")
