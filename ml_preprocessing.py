import glob, os
import pandas as pd
from pprint import pprint
from utils import load_yaml, FindCreateDirectory
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# saving the ML model to pickle file and load it
import pickle
import joblib

config_data = load_yaml()


def export_label_data(df_full):
    """
    Arg
    :return:
    Label data
    """
    class_to_evaluate = config_data.get("class_name_train")
    label_data = df_full[class_to_evaluate]
    # svm can handle string data
    if config_data.get("train_kind") == "svm" or config_data.get("train_kind") == "grid_svm":
        label_data = label_data
        print(label_data.head())
    # Tensorflow can handle numpy ndarray arrays
    elif config_data.get("train_kind") == "deep_learning":
        lb_encoder = LabelEncoder()
        label_data = lb_encoder.fit_transform(label_data)
        label_data = to_categorical(label_data)
        print(label_data[:5])
    # some sklearn ML models can handle numerical values on target class
    elif config_data.get("train_kind") == "supervised_lb":
        lb_encoder = LabelEncoder()
        label_data = lb_encoder.fit_transform(label_data)
        print(label_data[:5])
    # print the type if the labeled data
    print("Type of the labeled data:", type(label_data))
    return label_data


def remove_unnecessary_columns(df):
    """

    :param df:
    :return:
    """
    # remove unnecessary columns that will not be exploited by the training phase
    columns_to_remove = config_data.get("remove_columns")
    # append the targeted class (i.e. the y values)
    columns_to_remove.append(config_data.get("class_name_train"))
    print("Columns that will be removed::", columns_to_remove)
    df_feats_ml = df.drop(labels=columns_to_remove, axis=1)
    print("DF with ML features only:")
    print(df_feats_ml.shape)
    print("Print which columns remained for further processing (useful features):")
    print(df_feats_ml.select_dtypes(include=["object"]).columns)
    return df_feats_ml


def enumerate_categorical_values(df_feats_ml):
    """

    :param df_feats_ml:
    :return:
    """
    print("DF categorical columns:")
    print(df_feats_ml.select_dtypes(include=["object"]).columns)
    df_cat = df_feats_ml[config_data.get("enumeration_columns")]
    df_cat_oh = pd.get_dummies(df_cat)
    print("One hot transformed columns:")
    print(df_cat_oh.columns)
    print("One hot columns info:")
    df_feats_ml.drop(labels=config_data.get("enumeration_columns"), axis=1, inplace=True)
    df_feats_ml_num_oh = pd.concat([df_feats_ml, df_cat_oh], axis=1)
    print(df_feats_ml_num_oh.shape)
    print("Print if there are columns with object dtype:")
    print(df_feats_ml_num_oh.select_dtypes(include=["object"]).columns)
    return df_feats_ml_num_oh


def pipeline_numerical(df_num_attr):
    """
    Todo: Pipeline
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("std_scaler", StandardScaler),
        ("pca_reducer", PCA(n_components=config_data.get("pca_n_components"))),
    ])

    data_ml = pipeline.fit_transform(df_num_attr)
    return data_ml


def scaling(feat_data):
    scaler = StandardScaler()
    scaler.fit(feat_data)
    feat_data_normalized = scaler.transform(feat_data)

    # save the scaler
    exports_dir = FindCreateDirectory(config_data.get("exports_directory")).inspect_directory()
    scaler_save_path = os.path.join(exports_dir, "scaler.pkl")
    pickle.dump(scaler, open(scaler_save_path, "wb"))

    return feat_data_normalized


def dimensionality_reduction(feat_data):
    pca = PCA(n_components=config_data.get("pca_n_components"))
    pca.fit(feat_data)
    feat_data_pca = pca.transform(feat_data)
    # save the pca transformer
    exports_dir = FindCreateDirectory(config_data.get("exports_directory")).inspect_directory()
    pca_save_path = os.path.join(exports_dir, "pca_transformer.pkl")
    pickle.dump(pca, open(pca_save_path, "wb"))

    return feat_data_pca


def split_to_train_test(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test
