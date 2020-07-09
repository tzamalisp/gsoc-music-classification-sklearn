import os
from utils import load_yaml, FindCreateDirectory, LogsDeleter
from ml_load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from ml_load_low_level import FeaturesDf
from utils import DfChecker
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
        features_transformed = Transform(config=config_data, df=features, process="gaussianized").post_processing()
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

        # Model evaluation
        print("MODEL EVALUATION")
        eval_model = Evaluation(config=config_data,
                                model=model_trained,
                                x_data=features_transformed,
                                y_data=labels,
                                class_name=class_to_model)
        eval_model.model_evaluation()


def project_acousticbrainz():
    """

    :return:
    """


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
