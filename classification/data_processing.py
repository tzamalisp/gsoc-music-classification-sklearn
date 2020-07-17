import random
from termcolor import colored
from transformation.load_groung_truth import DatasetDFCreator
from transformation.features_labels import FeaturesLabelsSplitter


class DataProcessing:
    def __init__(self, config, dataset, class_name):
        self.config = config
        self.dataset = dataset
        self.class_name = class_name

        self.dataset_np_array = []

    def shuffle_tracks_data(self):
        print("Transform to numpy array:")
        # X_array = dataset_merge.values
        # dataset_np_array = X_array.tolist()

        print(colored("Type of the dataset inserted before shuffling: {}".format(type(self.dataset)), "green"))
        print(self.dataset[0])
        print(self.dataset[4])

        # Shuffling
        self.dataset_np_array = self.dataset
        print("Shuffle the data:")
        random.seed(a=self.config.get("random_seed"))
        random.shuffle(self.dataset_np_array)
        print("Check some indexes:")
        print(self.dataset_np_array[0])
        print(self.dataset_np_array[4])
        print("Shuffle array length: {}".format(len(self.dataset_np_array)))

        return self.dataset_np_array

    def exporting_classification_data(self):

        # create DF with the features, labels, and tracks together
        df_tracks_features_labels = DatasetDFCreator(self.config,
                                                     self.dataset_np_array,
                                                     self.class_name
                                                     ).create_df_tracks()
        print("Counted columns in the full shuffled df (target class + features): {}"
              .format(len(df_tracks_features_labels.columns)))
        print(df_tracks_features_labels.columns)
        print(df_tracks_features_labels[["track", self.class_name]].head())

        print()
        print("Exporting X and y..")
        # Export features from the DF
        X = FeaturesLabelsSplitter(config=self.config,
                                   df=df_tracks_features_labels,
                                   train_class=self.class_name
                                   ).export_features()
        # Export labels from the DF
        y = FeaturesLabelsSplitter(config=self.config,
                                   df=df_tracks_features_labels,
                                   train_class=self.class_name
                                   ).export_labels()

        print("Columns: {}".format(X.columns))

        return X, y
