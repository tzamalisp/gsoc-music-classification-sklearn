import os
import joblib
import json
import pandas as pd
from utils import load_yaml
from transformation.utils_preprocessing import flatten_dict_full
from transformation.transform_predictions import TransformPredictions


class Predict:
    def __init__(self, config, track_low_level, class_name):
        self.config = config
        self.track_low_level = track_low_level
        self.class_name = class_name

        self.exports_dir = ""
        self.best_model = ""
        self.track_feats = dict()

        self.load_best_model()
        # self.flat_dict()
        self.df_track = pd.DataFrame()
        self.list_track = []

    def load_best_model(self):
        self.exports_dir = os.path.join(os.getcwd(), "{}_{}".format(self.config["exports_directory"], self.class_name))
        best_model_path = os.path.join(self.exports_dir, "best_model_{}.json".format(self.class_name))
        # best_model_path = os.path.join(self.exports_dir, "models", "model_grid_{}.pkl".format[""])
        with open(best_model_path) as json_file:
            self.best_model = json.load(json_file)
        print("Best model:")
        pprint(self.best_model)

    def preprocessing(self):
        print("FLATTENING:")
        try:
            if 'beats_position' in self.track_low_level['rhythm']:
                del self.track_low_level['rhythm']['beats_position']
        except Exception as e:
            print("There is no 'rhythm' key in the low level data. Exception:", e)

        # data dictionary transformed to a fully flattened dictionary
        self.track_feats = dict(flatten_dict_full(self.track_low_level))
        list_track = []
        list_track.append(self.track_feats)
        print("DICT TO DATAFRAME:")
        self.df_track = pd.DataFrame(data=list_track, columns=list_track[0].keys())
        print("TYPE:", type(self.df_track))
        # print(self.df_track)
        # print("Shape of DF", self.df_track.shape)

        print("PROCESSING:")
        features_prepared = TransformPredictions(config=self.config,
                                                 df_feats=self.df_track,
                                                 process=self.best_model["preprocessing"],
                                                 exports_path=self.exports_dir
                                                 ).post_processing()
        print(features_prepared.shape)

        model_path = os.path.join(self.exports_dir, "models", "model.pkl")
        best_model_path = os.path.join(self.exports_dir, "models", "model_grid_{}.pkl".format(self.best_model["preprocessing"]))
        clf_loaded = joblib.load(best_model_path)
        predicted = clf_loaded.predict(features_prepared)
        predicted_prob = clf_loaded.predict_proba(features_prepared)
        print("Prediction:", predicted)
        print("Classes: ", clf_loaded.classes_)
        print("Prediction probabilities", predicted_prob)


if __name__ == '__main__':
    import requests
    from pprint import pprint
    config_data = load_yaml("configuration.yaml")
    # pprint(config_data)

    # "Idle Up" by Dousk & JMP - danceable
    response = requests.get('https://acousticbrainz.org/api/v1/78281677-8ba1-41df-b0f7-df6b024caf13/low-level')
    # "Shock the Monkey (Gorilla mix)" by Coal Chamber - danceable
    #  response = requests.get('https://acousticbrainz.org/api/v1/34959b3b-634f-415e-8fb8-b47a8e27e0e5/low-level')

    # "So Dear to My Heart" by Peggy Lee - not danceable
    # response = requests.get('https://acousticbrainz.org/api/v1/7fb1b586-017c-4a89-b15a-0bb837983108/low-level')
    # "See the Light" by Earth, Wind & Fire - not danceable
    # response = requests.get('https://acousticbrainz.org/api/v1/c129e3f4-3653-467a-a67f-c33bc912e6cb/low-level')
    track = response.json()
    # save the low level locally
    # with open("sample_track.json", "w") as outfile:
    #     json.dump(track, outfile, indent=4)

    if track["metadata"]["tags"]["artist"][0]:
        print("Artist:", track["metadata"]["tags"]["artist"][0])
    if track["metadata"]["tags"]["album"][0]:
        print("Track:", track["metadata"]["tags"]["album"][0])

    prediction = Predict(config=config_data,
                         track_low_level=track,
                         class_name="danceability")
    prediction.preprocessing()


