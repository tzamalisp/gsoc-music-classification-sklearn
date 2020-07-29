import os
import yaml
import pandas as pd
from pprint import pprint
from termcolor import colored
import random
from utils import load_yaml, FindCreateDirectory
from transformation.load_low_level import FeaturesDf


class ListGroundTruthFiles:
    """

    """
    def __init__(self, config):
        """

        :param config:
        """
        self.config = config
        self.dataset_dir = ""
        self.class_dir = ""

    def list_gt_filenames(self):
        """

        :return:
        """
        self.dataset_dir = self.config.get("ground_truth_directory")
        self.class_dir = self.config.get("class_dir")
        path = os.path.join(os.getcwd(), self.dataset_dir, self.class_dir, "metadata")
        ground_truth_list = [filename for filename in os.listdir(os.path.join(path))
                             if filename.startswith("groundtruth")]
        return ground_truth_list


class GroundTruthLoad:
    """
        The Ground Truth data object which contains features to:
         * counter the JSON low-level data
         * Todo: create logger object

         Attributes:
        """
    def __init__(self, config, gt_filename):
        """

        :param config:
        :param gt_filename:
        """
        self.config = config
        self.gt_filename = gt_filename
        self.class_dir = ""
        self.ground_truth_data = {}
        self.labeled_tracks = {}
        self.train_class = ""
        self.dataset_dir = ""
        self.tracks = []

        self.load_local_ground_truth()

    def load_local_ground_truth(self):
        """
        Loads the the ground truth file.
        * The directory with the dataset should be located inside the app folder location.
        :return:
        """
        self.dataset_dir = self.config.get("ground_truth_directory")
        self.class_dir = self.config.get("class_dir")
        with open(os.path.join(os.getcwd(), "{}/{}/metadata/{}".format(
                self.dataset_dir, self.class_dir, self.gt_filename)), "r") as stream:
            try:
                self.ground_truth_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def export_train_class(self):
        """

        :return:
        """
        self.train_class = self.ground_truth_data["className"]
        print("EXPORT CLASS NAME:", self.train_class)
        return self.train_class

    def export_gt_tracks(self):
        self.labeled_tracks = self.ground_truth_data["groundTruth"]
        print("GROUND TRUTH DICTIONARY LENGTH:", len(self.labeled_tracks))
        tracks_list = []
        for track, label in self.labeled_tracks.items():
            tracks_list.append((track, label))
        print(colored("SEED is set to: {}".format(self.config.get("seed"), "cyan")))
        random.seed(a=self.config.get("seed"))
        random.shuffle(tracks_list)
        return tracks_list

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
        for root, dirs, files in os.walk(os.path.join(os.getcwd(), self.dataset_dir)):
            for file in files:
                if file.endswith(".json"):
                    # print(os.path.join(root, file))
                    counter += 1
        print("counted json files:", counter)


class DatasetExporter:
    def __init__(self, config, tracks_list, train_class):
        self.config = config
        self.tracks_list = tracks_list
        self.train_class = train_class
        self.dataset_dir = ""
        self.class_dir = ""
        self.df_tracks = pd.DataFrame()
        self.df_feats = pd.DataFrame()
        self.y = []

    def create_df_tracks(self):
        """
        Creates the pandas DataFrame with the tracks.
        Todo: more comments
        :return:
        DataFrame or None: a DataFrame with the tracks included in the ground truth yaml file containing the track name,
        the path to load the JSON low-level data, the label, etc. Else, it returns None.
        """
        # the class name from the ground truth data that is the target
        self.dataset_dir = self.config.get("ground_truth_directory")
        self.class_dir = self.config.get("class_dir")
        print('DATASET-DIR', self.dataset_dir)
        print('CLASS NAME PATH', self.class_dir)
        # the path to the "features" directory that contains the rest of the low-level data sub-directories
        path_features = os.path.join(os.getcwd(), self.dataset_dir, self.class_dir, "features")
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
        path_low_level = os.path.join(os.getcwd(), self.dataset_dir, self.class_dir, "features", low_level_dir)
        print("Path of low level data: {}".format(path_low_level))
        # create a list with dictionaries that contain the information from each track in
        if low_level_dir != "":
            self.df_tracks = pd.DataFrame(data=self.tracks_list, columns=["track", self.train_class])
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
                print("There are no NULL values found.")

            # export shuffled tracks to CSV format
            tracks_csv_dir = os.path.join("{}_{}".format(self.config.get("exports_directory"),
                                                         self.train_class), "tracks_csv_format")
            tracks_csv_path = FindCreateDirectory(tracks_csv_dir).inspect_directory()
            self.df_tracks.to_csv(os.path.join(tracks_csv_path, "tracks_{}_shuffled.csv".format(self.train_class)))
            print("DF INFO:")
            print(self.df_tracks.info())
            print("COLUMNS CONTAIN OBJECTS", self.df_tracks.select_dtypes(include=['object']).columns)

            self.df_feats = FeaturesDf(df_tracks=self.df_tracks,
                                       train_class=self.train_class,
                                       path_low_level=path_low_level, config=self.config
                                       ).create_low_level_df()

            self.y = self.df_tracks[self.train_class].values

            return self.df_feats, self.y, self.df_tracks["track"].values
        else:
            return None, None, None


