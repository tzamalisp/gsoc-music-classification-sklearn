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
from ml_model import Evaluation


def main():
    config_data = load_yaml()

    df_gt_data = GroundTruthLoad().create_df_tracks()
    print()
    print()

    df_full = FeaturesDf(df_tracks=df_gt_data).concatenate_dfs()
    print(df_full.head())
    print(df_full.shape)

    y = export_label_data(df_full)
    print(type(y))

    # remove no-useful columns
    df_ml = remove_unnecessary_columns(df_full)
    # enumerate categorical data
    df_ml_num = enumerate_categorical_values(df_ml)
    # scale the data
    feats_scaled = scaling(df_ml_num)
    # pca apply
    feats_pca = dimensionality_reduction(feats_scaled)

    # labels (y)
    label_data = export_label_data(df_full)

    train_feats, test_feats, train_labels, test_labels = split_to_train_test(feats_pca, label_data)

    model_trained = TrainModel(config=config_data, features=feats_pca, labels=label_data).train_grid_search()

    eval_model = Evaluation(model=model_trained, x_data=test_feats, y_data=test_labels, model_name="SVM GridSearch")
    eval_model.model_evaluation()


def example(argument):
    """

    :param argument:
    :return:
    """
    value = argument
    return value


if __name__ == "__main__":
    main()

