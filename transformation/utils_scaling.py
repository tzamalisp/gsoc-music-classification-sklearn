import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sklearn scaling transformation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
import joblib

from transformation.utils_preprocessing import list_descr_handler, descr_selector
from utils import FindCreateDirectory


def descr_normalizing(feats_data, processing, config, exports_path, train_process, mode):
    """

    :param feats_data:
    :param processing:
    :param config:
    :param exports_path:
    :param train_process:
    :param mode:
    :return:
    """
    images_path = FindCreateDirectory(os.path.join(exports_path, "images")).inspect_directory()
    models_path = FindCreateDirectory(os.path.join(exports_path, "models")).inspect_directory()
    feats_data_columns = feats_data.columns
    print("length of df features columns: {}".format(len(feats_data_columns)))
    # normalize (fit/transform)
    if mode == "train":
        normalizer = MinMaxScaler()
        normalizer.fit(feats_data)
        feats_data_normalized = normalizer.transform(feats_data)
        joblib.dump(normalizer, os.path.join(models_path, "normalizer.pkl"))
        print("Type of normalized data: {}".format(type(feats_data_normalized)))
        feats_data = pd.DataFrame(data=feats_data_normalized, columns=feats_data_columns)
        print("Type of normalized data after conversion: {}".format(type(feats_data)))
        print("Normalization process completed.")
        # create normalized data distribution based on the first column data
        print(feats_data.iloc[:, 0].head())
        sns.distplot(feats_data.iloc[:, 0])
        # save the plot with the normalized data distribution
        print("Saving plot with the normalized data distribution")
        plt.savefig(os.path.join(images_path, "{}_normalized_data_distribution.png".format(train_process)))
        plt.close()
        # save the plot with the data depicted on a scatter plot
        sns.scatterplot(x=feats_data.iloc[:, 0], y=feats_data.iloc[:, 1], data=feats_data)
        plt.savefig(os.path.join(images_path, "{}_normalized_data_scatterplot.png".format(train_process)))
        plt.close()
        print()
    elif mode == "predict":
        normalizer = joblib.load(os.path.join(models_path, "normalizer.pkl"))
        print("Normalizer loaded..")
        feats_data_normalized = normalizer.transform(feats_data)
        print("Type of normalized data: {}".format(type(feats_data_normalized)))
        feats_data = pd.DataFrame(data=feats_data_normalized, columns=feats_data_columns)
    return feats_data


def descr_gaussianizing(feats_data, processing, config, exports_path, train_process, mode):
    images_path = FindCreateDirectory(os.path.join(exports_path, "images")).inspect_directory()
    models_path = FindCreateDirectory(os.path.join(exports_path, "models")).inspect_directory()
    feats_data_columns = feats_data.columns
    select_list = list_descr_handler(processing["params"]["descriptorNames"])
    print("Selection list: {}".format(select_list))
    print("Input DF - no. of columns: {}".format(len(feats_data_columns)))
    df_gauss = descr_selector(df=feats_data, descr_select_list=select_list)
    df_gauss_columns = df_gauss.columns
    print("Gaussian DF - no. of columns: {}".format(len(df_gauss_columns)))
    df_no_gauss = feats_data.drop(df_gauss_columns, axis=1)
    print("No Gaussian DF - no. of columns: {}".format(len(df_no_gauss.columns)))
    # gaussianize (fit/transform)
    if mode == "train":
        gaussianizer = QuantileTransformer(n_quantiles=1000)
        gaussianizer.fit(df_gauss)
        feats_data_gaussianized = gaussianizer.transform(df_gauss)
        joblib.dump(gaussianizer, os.path.join(models_path, "gaussianizer.pkl"))
        print("Type of gaussianized data: {}".format(type(feats_data_gaussianized)))
        feats_data_gaussianized = pd.DataFrame(data=feats_data_gaussianized, columns=df_gauss_columns)
        feats_data = pd.concat([feats_data_gaussianized, df_no_gauss], axis=1)
        print("Output DF - no. of columns: {}".format(len(feats_data.columns)))
        print("Gaussianization process completed.")
        # create gaussianized data distribution based on the first column data
        print(feats_data.iloc[:, 0].head())
        sns.distplot(feats_data.iloc[:, 0])
        # save the plot with the gaussianized data distribution
        print("Saving plot with the gaussianized data distribution")
        plt.savefig(os.path.join(images_path, "{}_gaussianized_data_distribution.png".format(train_process)))
        plt.close()
        # save the plot with the data depicted on a scatter plot
        sns.scatterplot(x=feats_data.iloc[:, 0], y=feats_data.iloc[:, 1], data=feats_data)
        plt.savefig(os.path.join(images_path, "{}_gaussianized_data_scatterplot.png".format(train_process)))
        plt.close()
        print()
    #  gaussianize (transform)
    elif mode == "predict":
        gaussianizer = joblib.load(os.path.join(models_path, "gaussianizer.pkl"))
        print("Gaussianizer loaded..")
        feats_data_gaussianized = gaussianizer.transform(df_gauss)
        print("Type of gaussianized data: {}".format(type(feats_data_gaussianized)))
        feats_data_gaussianized = pd.DataFrame(data=feats_data_gaussianized, columns=df_gauss_columns)
        feats_data = pd.concat([feats_data_gaussianized, df_no_gauss], axis=1)

    return feats_data


def descr_dimensionality_reduction(feats_data, processing, config, exports_path, train_process, mode):
    images_path = FindCreateDirectory(os.path.join(exports_path, "images")).inspect_directory()
    models_path = FindCreateDirectory(os.path.join(exports_path, "models")).inspect_directory()

    # TODO: apply pca
    if mode == "train":
        pca = PCA(config["pca_n_components"])
        pass
    elif mode == "predict":
        pass
    pass