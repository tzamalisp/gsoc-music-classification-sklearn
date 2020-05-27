from ml_load_groung_truth import GroundTruthLoad
from ml_load_low_level import FeaturesDf
from utils import DfChecker
from ml_preprocessing import export_label_data
from ml_preprocessing import remove_unnecessary_columns
from ml_preprocessing import enumerate_categorical_values
from ml_preprocessing import scaling
from ml_preprocessing import dimensionality_reduction

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


def main():
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

    # define the length of parameters
    parameters_grid = {'kernel': ['poly', 'rbf'],
                       'C': [0.1, 1, 10, 100, 3, 5, 7, 9, 11],
                       'gamma': [1, 0.1, 0.01, 0.001, 3, 5, 7, 9, 11],
                       'class_weight': ['balanced', None]
                       }

    svm = SVC(gamma="auto", probability=True)
    gsvc_pca = GridSearchCV(estimator=svm, param_grid=parameters_grid, cv=5)
    gsvc_pca.fit(feats_pca, label_data)
    print(gsvc_pca.best_score_)


if __name__ == "__main__":
    main()

