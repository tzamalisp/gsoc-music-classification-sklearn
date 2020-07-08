import os
from utils import load_yaml, FindCreateDirectory, LogsDeleter
from ml_load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from ml_load_low_level import FeaturesDf
from utils import DfChecker
from ml_preprocessing import remove_unnecessary_columns
from ml_preprocessing import enumerate_categorical_values
from ml_preprocessing import scaling
from ml_preprocessing import dimensionality_reduction
from ml_preprocessing import split_to_train_test
from classification.train_model import TrainModel
from transformation.transform import Transform
from transformation.features_labels import FeaturesLabelsSplitter
from classification.evaluation import Evaluation
import time
from datetime import datetime


def project_ground_truth():
    """

    :return:
    """
    config_data = load_yaml()
    if config_data["gaia_imitation"] is True:
        print("GAIA IMITATION MODE is ON")
    print("Dataset/class for evaluation:", config_data.get("class_dir"))
    print("Kind of training:", config_data.get("train_kind"))
    print()
    gt_files_list = ListGroundTruthFiles(config_data).list_gt_filenames()
    print(gt_files_list)
    print("LOAD GROUND TRUTH")
    for gt_file in gt_files_list:
        print("YAML FILE TO PROCESS:", gt_file)
        gt_data = GroundTruthLoad(config_data, gt_file)
        individual_df_gt_data = gt_data.create_df_tracks()
        class_to_model = gt_data.export_class_name()
        print("CLASS TO TRAIN AND IMPORT TO PROCESSING:", class_to_model)
        # delete logs if set True in config file
        log_deleter = LogsDeleter(config=config_data, train_class=class_to_model)
        log_deleter.delete_logs()

        print()
        print()
        model_training(df_gt_data=individual_df_gt_data, class_train=class_to_model, config=config_data)
        print()
        print()
        print()
        print()


def data_handling():
    config_data = load_yaml()
    if config_data["gaia_imitation"] is True:
        print("GAIA IMITATION MODE is ON")
    gt_files_list = ListGroundTruthFiles(config_data).list_gt_filenames()
    print(gt_files_list)
    print("LOAD GROUND TRUTH")
    for gt_file in gt_files_list:
        print("YAML file to load the data:", gt_file)
        gt_object = GroundTruthLoad(config_data, gt_file)
        gt_data = gt_object.create_df_tracks()
        class_to_model = gt_object.export_class_name()
        print("CLASS TO TRAIN AND IMPORT TO PROCESSING:", class_to_model)
        print("LOAD LOW LEVEL and FLATTEN THEM")
        df_full = FeaturesDf(df_tracks=gt_data, class_name=class_to_model, config=config_data).concatenate_dfs()
        feats_labels_splitter = FeaturesLabelsSplitter(config=config_data, df=df_full, train_class=class_to_model)
        # labels vs. features split
        labels = feats_labels_splitter.export_labels()
        features = feats_labels_splitter.export_features()
        # transformation
        features_transformed = Transform(config=config_data, df=features, process="mfcc").post_processing()
        print(features_transformed.columns)
        # Model train
        print("MODEL TRAINING")
        print("Starting training time calculation..")
        start = time.time()
        print("Start time: {}".format(datetime.now()))
        training = TrainModel(config=config_data,
                              train_data=features_transformed,
                              label_data=labels,
                              train_class=class_to_model)
        model_trained = training.train_model()
        print()
        end = time.time()
        print("End time: {}".format(datetime.now()))
        print("Train time: {} seconds".format(end - start))


def project_acousticbrainz():
    """

    :return:
    """


def model_training(df_gt_data, class_train, config):
    """

    :param df_gt_data:
    :param class_train:
    :param config:
    :return:
    """

    # print("DF full:")
    # print(df_full.head())
    # print("DF full shape: {}".format(df_full.shape))
    print()
    print()
    # labels (y)
    print("EXPORT LABEL/TARGET DATA (str, one-hot, encoded)")
    # label_data = export_label_data(df=df_full, class_name=class_train, config=config)
    # print("Type of target data:", type(label_data))
    print()
    print()

    # remove no-useful columns
    print("REMOVE NO USEFUL FEATURES")
    # df_ml = remove_unnecessary_columns(df=df_full, class_name=class_train, config=config)
    # print("SHAPE DF ML SHAPE: {}".format(df_ml.shape))
    # print("TYPE DF ML SHAPE: {}".format(type(df_ml)))
    print()
    print()

    # enumerate categorical data
    print("FEATURES ENUMERATION")
    # df_ml_num = enumerate_categorical_values(df_feats_ml=df_ml, config=config)
    print()
    print()

    # scale the data
    # feats_scaled = scaling(feat_data=df_ml_num, config=config)
    # print("SHAPE FEATS SCALED: {}".format(feats_scaled.shape))
    # print("TYPE FEATS SCALED: {}".format(type(feats_scaled)))
    print()
    print()

    # pca apply
    print("PCA")
    # feats_pca = dimensionality_reduction(feat_data=feats_scaled, config=config)
    # print("SHAPE FEATS PCA: {}".format(feats_pca.shape))
    # print("TYPE FEATS PCA: {}".format(type(feats_pca)))
    print()
    print()

    # train/test split
    print("TRAIN/TEST SPLIT")
    # train_feats, test_feats, train_labels, test_labels = split_to_train_test(x_data=feats_pca, y_data=label_data)

    print()
    print()




    print()
    print()

    # Model evaluation
    print("MODEL EVALUATION")
    # eval_model = Evaluation(config=config,
    #                         model=model_trained,
    #                         x_data=test_feats,
    #                         y_data=test_labels,
    #                         class_name=class_train)
    # eval_model.model_evaluation()



def example(argument):
    """

    :param argument:
    :return:
    """
    value = argument
    return value


if __name__ == "__main__":
    # project_ground_truth()
    data_handling()
