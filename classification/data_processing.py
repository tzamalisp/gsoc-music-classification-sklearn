import random
from transformation.load_groung_truth import DatasetDFCreator
from transformation.features_labels import FeaturesLabelsSplitter


class DataProcessing:
    def __init__(self, config, dataset, class_name):
        self.config = config
        self.dataset = dataset
        self.class_name = class_name

    def exporting_classification_data(self):
        print("Transform to numpy array:")
        # X_array = dataset_merge.values
        # dataset_np_array = X_array.tolist()

        print("Type of the dataset inserted before shuffling:", type(self.dataset))
        print(self.dataset[0])
        print(self.dataset[4])

        # Shuffling
        dataset_np_array = self.dataset
        print("Shuffle the data:")
        random.seed(a=self.config.get("random_seed"))
        random.shuffle(dataset_np_array)
        print("Check some indexes:")
        print(dataset_np_array[0])
        print(dataset_np_array[4])
        print("Shuffle array length: {}".format(len(dataset_np_array)))

        # create DF with the features, labels, and tracks together
        df_tracks_features_labels = DatasetDFCreator(self.config, dataset_np_array, "danceability").create_df_tracks()
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
