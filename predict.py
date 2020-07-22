import os
import joblib
import json
import pandas as pd
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from transformation.load_low_level import FeaturesDf
from transformation.utils_preprocessing import flatten_dict_full
from transformation.transform import Transform


class Predict:
    def __init__(self, config, track_low_level, process, class_name):
        self.config = config
        self.track_low_level = track_low_level
        self.process = process
        self.class_name = class_name

        self.exports_dir = ""
        self.best_model = ""
        self.track_feats = dict()

        self.load_best_model()
        self.flat_dict()

    def load_best_model(self):
        self.exports_dir = os.path.join(os.getcwd(), "{}_{}".format(self.config["exports_directory"], self.class_name))
        best_model_path = os.path.join(self.exports_dir, "best_model_{}.json".format(self.class_name))
        with open(best_model_path) as json_file:
            self.best_model = json.load(json_file)
        print("Best model:")
        pprint(self.best_model)

    def flat_dict(self):
        try:
            if 'beats_position' in self.track_low_level['rhythm']:
                del self.track_low_level['rhythm']['beats_position']
        except Exception as e:
            print("There is no 'rhythm' key in the low level data. Exception:", e)

        # data dictionary transformed to a fully flattened dictionary
        self.track_feats = dict(flatten_dict_full(self.track_low_level))
        list_track = []
        list_track.append(self.track_feats)
        self.df_track = pd.DataFrame(data=list_track, columns=list_track[0].keys())
        # print(self.df_track)

    def preprocessing(self):
        # transformation of the data
        X_transformed = Transform(config=self.config,
                                  df=self.df_track,
                                  process=self.best_model["preprocessing"],
                                  exports_path=self.exports_dir,
                                  mode="predict"
                                  ).post_processing()

        model_path = os.path.join(self.exports_dir, "models", "model.pkl")
        clf_loaded = joblib.load(model_path)
        predicted = clf_loaded.predict(X_transformed)
        predicted_prob = clf_loaded.predict_proba(X_transformed)
        print(predicted)
        print(clf_loaded.classes_)
        print(predicted_prob)


if __name__ == '__main__':
    import requests
    from pprint import pprint
    config_data = load_yaml("configuration.yaml")
    # pprint(config_data)
    # response = requests.get('https://acousticbrainz.org/api/v1/7fb1b586-017c-4a89-b15a-0bb837983108/low-level')
    # jmp
    response = requests.get('https://acousticbrainz.org/api/v1/78281677-8ba1-41df-b0f7-df6b024caf13/low-level')
    # Earth, Wind & Fire
    # response = requests.get('https://acousticbrainz.org/api/v1/c129e3f4-3653-467a-a67f-c33bc912e6cb/low-level')
    track = response.json()
    if track["metadata"]["tags"]["artist"][0]:
        print("Artist:", track["metadata"]["tags"]["artist"][0])
    if track["metadata"]["tags"]["album"][0]:
        print("Track:", track["metadata"]["tags"]["album"][0])

    prediction = Predict(config=config_data,
                         track_low_level=track,
                         process="normalized",
                         class_name="danceability")
    prediction.preprocessing()
