import os
import requests
import argparse
from pprint import pprint
import joblib
import json
import pandas as pd
from utils import load_yaml, FindCreateDirectory
from transformation.utils_preprocessing import flatten_dict_full
from transformation.transform_predictions import TransformPredictions


class Predict:
    def __init__(self, config, track_low_level):
        self.config = config
        self.track_low_level = track_low_level

        self.class_name = ""
        self.exports_path = ""
        self.exports_dir = ""
        self.best_model = ""
        self.track_feats = dict()

        self.load_best_model()
        # self.flat_dict()
        self.df_track = pd.DataFrame()
        self.list_track = []

    def load_best_model(self):
        self.class_name = self.config["class_name"]
        self.exports_path = self.config["exports_path"]
        self.exports_dir = "{}_{}".format(self.config["exports_directory"], self.class_name)

        # self.exports_path = os.path.join(self.exports_path, "{}_{}".format(self.exports_dir, self.class_name))
        best_model_path = os.path.join(self.exports_path,
                                       self.exports_dir,
                                       "best_model_{}.json".format(self.class_name))
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
                                                 train_class=self.class_name,
                                                 exports_path=self.exports_path
                                                 ).post_processing()
        print(features_prepared.shape)
        models_path = FindCreateDirectory(self.exports_path,
                                          os.path.join(self.exports_dir, "models")).inspect_directory()
        best_model_path = os.path.join(models_path, "model_grid_{}.pkl".format(self.best_model["preprocessing"]))
        clf_loaded = joblib.load(best_model_path)
        predicted = clf_loaded.predict(features_prepared)
        predicted_prob = clf_loaded.predict_proba(features_prepared)
        print("Prediction:", predicted)
        print("Classes: ", clf_loaded.classes_)
        print("Prediction probabilities", predicted_prob)


def prediction(exports_path, project_file, track_api):
    # if empty, path is declared as the app's main directory

    if exports_path is None:
        exports_path = os.getcwd()

    try:
        project_data = load_yaml("{}.yaml".format(project_file))
    except Exception as e:
        print('Unable to open project configuration file:', e)
        raise

    response = requests.get(track_api)

    track = response.json()
    if track["metadata"]["tags"]["artist"][0]:
        print("Artist:", track["metadata"]["tags"]["artist"][0])
    if track["metadata"]["tags"]["album"][0]:
        print("Track:", track["metadata"]["tags"]["album"][0])

    prediction_track = Predict(config=project_data,
                               track_low_level=track
                               )
    prediction_track.preprocessing()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Predictions.')

    parser.add_argument('-p', '--path',
                        dest="exports_path",
                        help='Path where the project results are stored.')

    parser.add_argument('-f', '--file',
                        dest="project_file",
                        help='Name prefix of the project configuration file (.yaml) that is stored.',
                        required=True)

    parser.add_argument('-t', '--track',
                        dest="track_api",
                        help='Low-level data link from the AcousticBrainz API.',
                        required=True)

    args = parser.parse_args()

    prediction(exports_path=args.exports_path,
               project_file=args.project_file,
               track_api=args.track_api)
