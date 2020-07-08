import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


class FeaturesLabelsSplitter:
    def __init__(self, config, df, train_class):
        self.config = config
        self.df = df
        self.train_class = train_class
        self.label_data = pd.Series()

    def export_labels(self):
        """

        :param df:
        :param class_name:
        :param config:
        :return:
        """
        print("Export target class/labels:", self.train_class)
        self.label_data = self.df[self.train_class]
        # svm can handle string data
        if self.config.get("train_kind") == "svm" or self.config.get("train_kind") == "grid_svm":
            self.label_data = self.label_data
            print("Label Data:")
            print(self.label_data.head())
            print("Unique labels - values:\n", self.label_data.value_counts())
        # TensorFlow can handle numpy ndarray arrays
        elif self.config.get("train_kind") == "deep_learning":
            lb_encoder = LabelEncoder()
            self.label_data = lb_encoder.fit_transform(self.label_data)
            self.label_data = to_categorical(self.label_data)
            print(self.label_data[:5])
            print("Shape of categorical data:", self.label_data.shape)
        # some sklearn ML models can handle numerical values on target class
        elif self.config.get("train_kind") == "supervised_lb":
            lb_encoder = LabelEncoder()
            label_data = lb_encoder.fit_transform(self.label_data)
            print(label_data[:5])

        # print the type if the labeled data
        print("Type of the labeled data:", type(self.label_data))
        return self.label_data

    def export_features(self):
        print("Export features..")
        features = self.df.drop(labels=[self.train_class], axis=1)
        print("Features shape: {}".format(features.shape))
        print("Features type: {}".format(type(features)))
        return features
