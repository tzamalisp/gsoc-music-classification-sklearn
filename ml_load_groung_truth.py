import os
import yaml
import pandas as pd
from pprint import pprint
from utils import DfChecker
from utils import load_yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class GroundTruthLoad:
    """
        The Ground Truth data object which contains features to:
         * counter the JSON low-level data
         * Todo: create logger object

         Attributes:
        """
    def __init__(self, config):
        """
        Todo: description
        """
        self.config = config
        self.class_to_evaluate = ""
        self.path_app = ""
        self.class_name_path = ""
        self.ground_truth_data = {}
        self.labeled_tracks = {}
        self.class_name = ""
        self.dataset_dir = ""
        self.tracks = []
        self.df_tracks = pd.DataFrame()

        self.load_local_ground_truth()
        # self.create_df_tracks()

    def get_path_app(self):
        """
        Finds the path of the folder that the application is located.
        :return:
        """
        self.path_app = os.path.abspath(os.getcwd())
        print("Current path:", self.path_app)
        print("Type:", type(self.path_app))

    def load_local_ground_truth(self):
        """
        Loads the the ground truth file.
        * The directory with the dataset should be located inside the app folder location.
        :return:
        """
        self.path_app = os.path.abspath(os.getcwd())
        self.dataset_dir = self.config.get("ground_truth_directory")
        self.class_to_evaluate = self.config.get("class_name_train")
        with open(os.path.join(self.path_app, "{}/{}/metadata/groundtruth.yaml".format(
                self.dataset_dir, self.class_to_evaluate)), "r") as stream:
            try:
                self.ground_truth_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def create_df_tracks(self):
        """
        Creates the pandas DataFrame with the tracks.
        Todo: more comments
        :return:
        DataFrame or None: a DataFrame with the tracks included in the ground truth yaml file containing the track name,
        the path to load the JSON low-level data, the label, etc. Else, it returns None.
        """
        self.labeled_tracks = self.ground_truth_data["groundTruth"]
        # the class name from the ground truth data that is the target
        self.class_name = self.ground_truth_data["className"]
        if self.class_name == "timbre":
            self.class_name_path = "timbre_bright_dark"
        else:
            self.class_name_path = self.class_name
        print('APP:', self.path_app)
        print('DATASET-DIR', self.dataset_dir)
        print('CLASS NAME PATH', self.class_name_path)
        # the path to the "features" directory that contains the rest of the low-level data sub-directories
        path_features = os.path.join(self.path_app, self.dataset_dir, self.class_name_path, "features")
        print("PATHHHH", path_features)
        # check if the "features" directory is empty or contains the "mp3" or the "orig" sub-directory
        low_level_dir = ""
        if len(os.listdir(path_features)) == 0:
            print("Directory is empty")
        else:
            print("Directory is not empty")
            directory_contents = os.listdir(path_features)
            if "mp3" in directory_contents:
                low_level_dir = "mp3"
            elif "orig" in directory_contents:
                low_level_dir = "orig"
            else:
                low_level_dir = ""
                print("There is no valid low-level data inside the features directory")
        # print which directory contains the low-level sub-directories (if exist)
        print("Low-level directory name that contains the data:", low_level_dir)
        # path to the low-level data sub-directories
        path_low_level = os.path.join(self.path_app, self.dataset_dir, self.class_name_path, "features", low_level_dir)
        # create a list with dictionaries that contain the information from each track in
        if low_level_dir != "":
            for key, value in self.labeled_tracks.items():
                track_dict = {}
                key = key.split("/")
                path_directory = key[:-1]
                path_directory_str = '/'.join(path_directory)
                track_json_name = key[-1]
                # print(key)
                path_tracks = os.path.join(path_low_level, path_directory_str)
                # print(path_tracks)
                for f_name in os.listdir(path_tracks):
                    if f_name.startswith(track_json_name):
                        track_dict["json_directory"] = path_directory_str
                        track_dict["track"] = track_json_name
                        track_dict["track_path"] = os.path.join(path_low_level, path_directory_str, f_name)
                        track_dict[self.class_name] = value
                self.tracks.append(track_dict)
            self.df_tracks = pd.DataFrame(data=self.tracks)
            print("Shape of tracks DF created before cleaning:", self.df_tracks.shape)
            print("Check the shape of a temporary DF that includes if there are any NULL values:")
            print(self.df_tracks[self.df_tracks.isnull().any(axis=1)].shape)
            print("Drop rows with NULL values if they exist..")
            if self.df_tracks[self.df_tracks.isnull().any(axis=1)].shape[0] != 0:
                self.df_tracks.dropna(inplace=True)
                print("Check if there are NULL values after the cleaning process:")
                print(self.df_tracks[self.df_tracks.isnull().any(axis=1)].shape)
                print("Re-index the tracks DF..")
                self.df_tracks = self.df_tracks.reset_index(drop=True)
            else:
                print("There are no NULL values values found.")
            self.df_tracks.to_csv(os.path.join(self.config.get("exports_directory"),
                                               "tracks_csv", "df_{}.csv".format(self.config.get("class_name_train"))))
            return self.df_tracks

        else:
            return None

    def check_ground_truth_data(self):
        """
        Todo: description
        :return:
        """
        pprint(self.ground_truth_data)

    def check_ground_truth_info(self):
        """
        Todo: description
        :return:
        """
        len(self.ground_truth_data["groundTruth"].keys())
        print("Ground truth data class/target:", self.ground_truth_data["className"])
        print("Label tracks:", type(self.labeled_tracks))
        print("Ground truth data keys - tracks:", len(self.ground_truth_data["groundTruth"].keys()))

    def check_tracks_folders(self):
        """
        Todo: function explanation docstring
        :return:
        """
        if len(self.labeled_tracks.keys()) is not 0:
            folders = []
            for key in self.labeled_tracks:
                key = key.split('/')
                path_sub_dir = '/'.join(key[:-1])
                folders.append(path_sub_dir)
            folders = set(folders)
            folders = list(folders)
            folders.sort()
            print("Directories that contain the low-level JSON data:")
            pprint(folders)

    def count_json_low_level_files(self):
        """
        Prints the JSON low-level data that is contained inside the dataset directory (the dataset
        directory is declared in configuration file).
        :return:
        """
        counter = 0
        for root, dirs, files in os.walk(os.path.join(self.path_app, self.dataset_dir)):
            for file in files:
                if file.endswith(".json"):
                    # print(os.path.join(root, file))
                    counter += 1
        print("counted json files:", counter)


if __name__ == "__main__":
    print("GROUND TRUTH LOAD SCRIPT")
    config_data = load_yaml()
    df_gt_data = GroundTruthLoad(config_data).create_df_tracks()
    print()
    print()

    # df GT data info
    print("CHECK THE GT DF INFO:")
    checker = DfChecker(df_check=df_gt_data)
    checker.check_df_info()
