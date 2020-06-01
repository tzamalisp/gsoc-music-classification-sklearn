from utils import load_yaml, FindCreateDirectory
from ml_load_groung_truth import GroundTruthLoad
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


def main():
    config_data = load_yaml()
    print("Dataset/class for evaluation:", config_data.get("class_name_train"))
    print("Kind of training:", config_data.get("train_kind"))
    print()
    print("LOAD GROUND TRUTH")
    df_gt_data = GroundTruthLoad(config_data).create_df_tracks()
    print()
    print()
    print("LOAD LOW LEVEL and FLATTEN THEM")
    df_full = FeaturesDf(df_tracks=df_gt_data).concatenate_dfs()
    print(df_full.head())
    print(df_full.shape)
    print()
    print()
    print("EXPORTING LABELS")
    y = export_label_data(df_full, config_data)
    print(type(y))
    print()
    print()

    # remove no-useful columns
    print("REMOVE NO USEFUL FEATURES")
    df_ml = remove_unnecessary_columns(df_full, config_data)
    print()
    print()

    # enumerate categorical data
    print("FEATURES ENUMERATION")
    df_ml_num = enumerate_categorical_values(df_ml, config_data)
    print()
    print()

    # scale the data
    print("SCALING")
    feats_scaled = scaling(df_ml_num, config_data)
    print()
    print()

    # pca apply
    print("PCA")
    feats_pca = dimensionality_reduction(feats_scaled, config_data)
    print()
    print()

    # labels (y)
    print("EXPORT LABEL DATA (str, one-hot, encoded)")
    label_data = export_label_data(df_full, config_data)
    print()
    print()

    # train/test split
    print("TRAIN/TEST SPLIT")
    train_feats, test_feats, train_labels, test_labels = split_to_train_test(feats_pca, label_data)

    print()
    print()

    # Model train
    print("MODEL TRAINING")
    print("Starting training time calculation..")
    start = time.time()
    print("Start time:", datetime.now())
    if config_data.get("train_kind") == "grid_svm":
        model_trained = TrainModel(config=config_data,
                                   features=feats_pca,
                                   labels=label_data
                                   ).train_grid_search()
    elif config_data.get("train_kind") == "svm":
        model_trained = TrainModel(config=config_data,
                                   features=feats_pca,
                                   labels=label_data
                                   ).train_svm()
    elif config_data.get("train_kind") == "deep_learning":
        model_trained = TrainModel(config=config_data,
                                   features=feats_pca,
                                   labels=label_data
                                   ).train_neural_network()
    print()
    end = time.time()
    print("End time:", datetime.now())
    print()
    print()

    # Model evaluation
    print("MODEL EVALUATION")
    eval_model = Evaluation(config=config_data,
                            model=model_trained,
                            x_data=test_feats,
                            y_data=test_labels)
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
    main()

