import os
from utils import load_yaml, FindCreateDirectory
from ml_load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from ml_load_low_level import FeaturesDf
from utils import DfChecker
from ml_preprocessing import export_label_data
from ml_preprocessing import remove_unnecessary_columns
from ml_preprocessing import enumerate_categorical_values
from ml_preprocessing import scaling
from ml_preprocessing import dimensionality_reduction
from ml_preprocessing import split_to_train_test
from ml_model import TrainModel
from ml_evaluation import Evaluation
import time
from datetime import datetime


def project_ground_truth():
    """

    :return:
    """
    config_data = load_yaml()
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
        # delete logs for specific model and class on a new run
        if config_data["delete_logs"] is True:
            dir_name = os.path.join(os.getcwd(), "evaluations")
            evaluations_list = os.listdir(dir_name)
            for item in evaluations_list:
                if item.endswith(".txt"):
                    if item.startswith("{}_{}".format(class_to_model, config_data["train_kind"])):
                        os.remove(os.path.join(dir_name, item))
            print("Previous evaluation deleted successfully.")

        print()
        print()
        model_training(df_gt_data=individual_df_gt_data, class_train=class_to_model, config=config_data)
        print()
        print()
        print()
        print()


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
    print("LOAD LOW LEVEL and FLATTEN THEM")
    df_full = FeaturesDf(df_tracks=df_gt_data).concatenate_dfs()
    print(df_full.head())
    print(df_full.shape)
    print()
    print()
    # labels (y)
    print("EXPORT LABEL/TARGET DATA (str, one-hot, encoded)")
    label_data = export_label_data(df=df_full, class_name=class_train, config=config)
    print("Type of target data:", type(label_data))
    print()
    print()

    # remove no-useful columns
    print("REMOVE NO USEFUL FEATURES")
    df_ml = remove_unnecessary_columns(df=df_full, class_name=class_train, config=config)
    print()
    print()

    # enumerate categorical data
    print("FEATURES ENUMERATION")
    df_ml_num = enumerate_categorical_values(df_feats_ml=df_ml, config=config)
    print()
    print()

    # scale the data
    print("SCALING")
    feats_scaled = scaling(feat_data=df_ml_num, config=config)
    print()
    print()

    # pca apply
    print("PCA")
    feats_pca = dimensionality_reduction(feat_data=feats_scaled, config=config)
    print()
    print()

    # train/test split
    print("TRAIN/TEST SPLIT")
    train_feats, test_feats, train_labels, test_labels = split_to_train_test(x_data=feats_pca, y_data=label_data)

    print()
    print()

    # Model train
    print("MODEL TRAINING")
    print("Starting training time calculation..")
    start = time.time()
    print("Start time:", datetime.now())
    if config.get("train_kind") == "grid_svm":
        model_trained = TrainModel(config=config,
                                   features=feats_pca,
                                   labels=label_data,
                                   class_name=class_train
                                   ).train_grid_search()
    elif config.get("train_kind") == "svm":
        model_trained = TrainModel(config=config,
                                   features=feats_pca,
                                   labels=label_data,
                                   class_name=class_train
                                   ).train_svm()
    elif config.get("train_kind") == "deep_learning":
        model_trained = TrainModel(config=config,
                                   features=feats_pca,
                                   labels=label_data,
                                   class_name=class_train
                                   ).train_neural_network()
    print()
    end = time.time()
    print("End time:", datetime.now())
    print()
    print()

    # Model evaluation
    print("MODEL EVALUATION")
    eval_model = Evaluation(config=config,
                            model=model_trained,
                            x_data=test_feats,
                            y_data=test_labels,
                            class_name=class_train)
    eval_model.model_evaluation()
    print()
    print("Train time:", end - start)


def example(argument):
    """

    :param argument:
    :return:
    """
    value = argument
    return value


if __name__ == "__main__":
    project_ground_truth()
