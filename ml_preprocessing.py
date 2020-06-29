import glob, os
import pandas as pd
from pprint import pprint
from utils import FindCreateDirectory
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# saving the ML model to pickle file and load it
import pickle
import joblib


def export_label_data(df, class_name, config):
    """

    :param df:
    :param class_name:
    :param config:
    :return:
    """
    print("EXPORT TARGET:", class_name)
    label_data = df[class_name]
    # svm can handle string data
    if config.get("train_kind") == "svm" or config.get("train_kind") == "grid_svm":
        label_data = label_data
        print("Label Data:")
        print(label_data.head())
        print("Unique labels - values:\n", label_data.value_counts())
    # TensorFlow can handle numpy ndarray arrays
    elif config.get("train_kind") == "deep_learning":
        lb_encoder = LabelEncoder()
        label_data = lb_encoder.fit_transform(label_data)
        label_data = to_categorical(label_data)
        print(label_data[:5])
        print("Shape of categorical data:", label_data.shape)
    # some sklearn ML models can handle numerical values on target class
    elif config.get("train_kind") == "supervised_lb":
        lb_encoder = LabelEncoder()
        label_data = lb_encoder.fit_transform(label_data)
        print(label_data[:5])

    # print the type if the labeled data
    print("Type of the labeled data:", type(label_data))
    return label_data


def remove_unnecessary_columns(df, class_name, config):
    """

    :param df:
    :param class_name:
    :param config:
    :return:
    """
    # remove unnecessary columns that will not be exploited by the training phase
    columns_to_remove = []
    # columns_to_remove.append(col for col in config["remove_columns"])
    for col in config["remove_columns"]:
        columns_to_remove.append(col)
    print("COLUMNS TO REMOVE BEFORE ADDING THE CLASS:", columns_to_remove)
    # append the targeted class (i.e. the y values)
    print("CLASS NAME TO REMOVE FROM THIS DF:", class_name)
    columns_to_remove.append(class_name)
    print("Columns that will be removed:", columns_to_remove)
    df_feats_ml = df.drop(labels=columns_to_remove, axis=1)
    print("DF with ML features only:")
    print(df_feats_ml.shape)
    print("Print which columns remained for further processing (useful features):")
    print(df_feats_ml.select_dtypes(include=["object"]).columns)
    return df_feats_ml


def enumerate_categorical_values(df_feats_ml, config):
    """

    :param df_feats_ml:
    :param config:
    :return:
    """
    print("DF categorical columns:")
    print(df_feats_ml.select_dtypes(include=["object"]).columns)
    df_cat = df_feats_ml[config.get("enumeration_columns")]
    df_cat_oh = pd.get_dummies(df_cat)
    print("One hot transformed columns:")
    print(df_cat_oh.columns)
    print("One hot columns info:")
    df_feats_ml.drop(labels=config.get("enumeration_columns"), axis=1, inplace=True)
    df_feats_ml_num_oh = pd.concat([df_feats_ml, df_cat_oh], axis=1)
    print(df_feats_ml_num_oh.shape)
    print("Print if there are columns with object dtype:")
    print(df_feats_ml_num_oh.select_dtypes(include=["object"]).columns)
    return df_feats_ml_num_oh


def pipeline_numerical(df_num_attr, config):
    """

    :param df_num_attr:
    :param config:
    :return:
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("std_scaler", StandardScaler),
        ("pca_reducer", PCA(n_components=config.get("pca_n_components"))),
    ])

    data_ml = pipeline.fit_transform(df_num_attr)
    return data_ml


def scaling(feat_data, config):
    """
    :param feat_data:
    :param config:
    :return:
    """
    if config.get("scaling") is "standard_scaled":
        scaler = StandardScaler()
    elif config.get("scaling") is "normalize_scaled":
        scaler = Normalizer()
    elif config.get("scaling") is "minmax_scaled":
        scaler = MinMaxScaler()
    elif config.get("scaling") is "robust_scaled":
        scaler = RobustScaler()
    else:
        # default value if nothing from the above is declared
        scaler = StandardScaler()

    # scaling data
    scaler.fit(feat_data)
    feat_data_normalized = scaler.transform(feat_data)

    # save the scaler
    exports_dir = FindCreateDirectory(config.get("exports_directory")).inspect_directory()
    scaler_save_path = os.path.join(exports_dir, "scaler.pkl")
    pickle.dump(scaler, open(scaler_save_path, "wb"))

    return feat_data_normalized


def dimensionality_reduction(feat_data, config):
    """

    :param feat_data:
    :param config:
    :return:
    """
    pca = PCA(n_components=config.get("pca_n_components"))
    pca.fit(feat_data)
    feat_data_pca = pca.transform(feat_data)
    print("Number of the most important features:", len(pca.singular_values_))
    # save the pca transformer
    exports_dir = FindCreateDirectory(config.get("exports_directory")).inspect_directory()
    pca_save_path = os.path.join(exports_dir, "pca_transformer.pkl")
    pickle.dump(pca, open(pca_save_path, "wb"))

    return feat_data_pca


def split_to_train_test(x_data, y_data):
    """

    :param x_data:
    :param y_data:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test
