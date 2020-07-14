#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Gaia
#
# Gaia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/

from __future__ import absolute_import
import os
# from gaia2.utils import TextProgress
# from .groundtruth import GroundTruth
# from .confusionmatrix import ConfusionMatrix
from pprint import pprint
import random
import logging
import pandas as pd
from utils import load_yaml, FindCreateDirectory
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from ml_load_groung_truth import ListGroundTruthFiles, GroundTruthLoad, DatasetDFCreator
from transformation.features_labels import FeaturesLabelsSplitter
from transformation.transform import Transform


# def shuffle_data(df_ml_data, config):
#     """
#
#     :param df_ml_data: (Pandas DataFrame) the data to be shuffled
#     :param config: (dict) the configuration data
#     :return: (NumPy array) the shuffled data
#     """
#     df_ml_cols = df_ml_data.columns
#     # convert DataFrame to NumPy array
#     ml_values = df_ml_data.values
#     # shuffle the data
#     random.seed(a=config.get("random_seed"))
#     random.shuffle(ml_values)
#     # convert the NumPy array to DF
#     df_ml_shuffle = pd.DataFrame(data=ml_values, columns=df_ml_cols)
#     return df_ml_shuffle

def display_scores(scores):
    print("Display scores:")
    print("Scores: {}".format(scores))
    print("Mean: {}".format(scores.mean()))
    print("Standard Deviation: {}".format(scores.std()))


def evaluate(classifier, dataset, groundTruth, confusion=None, nfold=None, verbose=True):
    """Evaluate the classifier on the given dataset and returns the confusion matrix.

    Uses only the points that are in the groundTruth parameter for the evaluation.

    Parameters
    ----------

    classifier  : a function which given a point returns its class
    dataset     : the dataset from which to get the points
    groundTruth : a map from the points to classify to their respective class
    """

    progress = TextProgress(len(groundTruth))
    done = 0

    confusion = confusion or ConfusionMatrix()

    for pointId, expected in groundTruth.items():
        try:
            found = classifier(dataset.point(pointId))
            if nfold is None:
                confusion.add(expected, found, pointId)
            else:
                confusion.addNfold(expected, found, pointId, nfold)

        except Exception as e:
            log.warning('Could not classify point "%s" because %s' % (pointId, str(e)))
            raise

        done += 1
        if verbose: progress.update(done)

    return confusion


def evaluateNfold(nfold, dataset, groundTruth, trainingFunc, config, seed=None, *args, **kwargs):
    """Evaluate the classifier on the given dataset and returns the confusion matrix.

    The evaluation is performed using n-fold cross validation.
    Uses only the points that are in the groundTruth parameter for the evaluation.

    Parameters
    ----------

    nfold        : the number of folds to use for the cross-validation
    dataset      : the dataset from which to get the points
    groundTruth  : a map from the points to classify to their respective class
    trainingFunc : a function which will train and return a classifier given a dataset,
                   the groundtruth, and the *args and **kwargs arguments
    """
    nfold = config["gaia_kfold_cv_n_splits"]
    log.info('Doing %d-fold cross validation' % nfold)

    print("Transform to numpy array:")
    # X_array = dataset_merge.values
    # X_array_list = X_array.tolist()

    print("Type of the dataset inserted before shuffling:", type(dataset))
    print(dataset[0])
    print(dataset[4])

    # Shuffling
    X_array_list = dataset
    print("Shuffle the data:")
    random.seed(a=config.get("random_seed"))
    random.shuffle(X_array_list)
    print("Check some indexes:")
    print(X_array_list[0])
    print(X_array_list[4])
    # pprint(X_array_list[:10])
    print("Shuffle array length: {}".format(len(X_array_list)))

    # create DF with the features, labels, and tracks together
    df_tracks_features = DatasetDFCreator(config, X_array_list, "danceability").create_df_tracks()
    print("Counted columns in the full shuffled df (target class + features): {}"
          .format(len(df_tracks_features.columns)))
    print(df_tracks_features.columns)
    print(df_tracks_features[["track", class_name]].head())

    print()
    print("Exporting X and y..")
    # Export features from the DF
    X = FeaturesLabelsSplitter(config=config, df=df_tracks_features, train_class=class_name).export_features()
    # Export labels from the DF
    y = FeaturesLabelsSplitter(config=config, df=df_tracks_features, train_class=class_name).export_labels()
    print("Columns: {}".format(X.columns))

    if config["gaia_imitation"] is True:
        gaia_params = load_yaml("gaia_best_models/jmp_results_{}.param".format(class_name))
        print("Gaia best model params: {}".format(gaia_params))

        # params data transformation
        preprocessing = gaia_params["model"]["preprocessing"]

        # params SVC
        C = 2 ** gaia_params["model"]["C"]
        gamma = 2 ** gaia_params["model"]["gamma"]
        kernel = gaia_params["model"]["kernel"].lower()
        balanceClasses = gaia_params["model"]["balanceClasses"]
        # TODO: declare a dictionary for class weights via automated labels balancing (unresponsive dataset)
        if balanceClasses is True:
            class_weights = "balanced"
        elif balanceClasses is False:
            class_weights = None
        else:
            print("Define a correct class weight value")




    # Transform dataset
    # pre-processing: data cleaning/enumerating/selecting descriptors
    # pre-processing: scaling
    print("Exports path for the training:")
    exports_dir = "{}_{}".format(config_data.get("exports_directory"), class_name)
    exports_path = FindCreateDirectory(exports_dir).inspect_directory()
    print(exports_path)
    X_transformed = Transform(config=config,
                              df=X,
                              process=preprocessing,
                              exports_path=exports_path
                              ).post_processing()
    print(X_transformed.columns)
    print(X_transformed.head())

    X_array_transformed = X_transformed.values

    from sklearn.svm import SVC
    svm = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weights,
        probability=config.get("svc_probability")
    )

    print("Evaluate the classifier with cross_val_score:")
    scores = cross_val_score(estimator=svm,
                             X=X_array_transformed,
                             y=y,
                             scoring="accuracy",
                             cv=nfold,
                             n_jobs=config.get("parallel_jobs"),
                             verbose=config.get("verbose")
                             )

    print()
    print("Score results:")
    display_scores(scores)
    print()
    print()




    # print("Folding..")
    # # Train the classifier with K-Fold cross-validation
    # random_seed = None
    # # shuffle = config["k_fold_shuffle"]
    # shuffle_f_fold = False
    # if shuffle_f_fold is True:
    #     random_seed = config["k_fold_shuffle"]
    # elif shuffle_f_fold is False:
    #     random_seed = None
    # print("Fitting the data to the classifier with K-Fold cross-validation..")
    # kf = KFold(n_splits=nfold,
    #            shuffle=shuffle_f_fold,
    #            random_state=random_seed
    #            )
    # print()
    # print()
    # # tracks_fold_indexing = []
    # tracks_fold_indexing_dict = {}
    # tracks_fold_indexing_list = []
    # print(X_array_list[0])
    # print(X_array_list[4])
    # fold_number = 0
    # for train_index, val_index in kf.split(X_array_list):
    #     print("Fold: {}".format(fold_number))
    #     # print("TRAIN INDEX: ", train_index)
    #     print("TEST INDEX: ", val_index)
    #     # print(len(train_index))
    #     print("Length of the test index array: {}".format(len(val_index)))
    #
    #     tracks_count = 0
    #     for index in val_index:
    #         # track = df_shuffled["folder_name"].iloc[index]
    #         track = X_array_list[index][0]
    #         # print(track)
    #         tracks_fold_indexing_dict[track] = fold_number
    #         tracks_fold_indexing_list.append("{}: {}".format(track, fold_number))
    #         tracks_count += 1
    #     print("tracks indexed to the specific fold:", tracks_count)
    #     fold_number += 1
    #
    # print()
    # print()
    # print("Dictionary:")
    # pprint(tracks_fold_indexing_dict)
    # print("length of keys:", len(tracks_fold_indexing_dict.keys()))
    # print()
    # print()


if __name__ == '__main__':
    log = logging.getLogger('classification.Evaluation')

    config_data = load_yaml("configuration.yaml")

    gt_data = GroundTruthLoad(config_data, "groundtruth.yaml")
    df_fg_data = gt_data.export_gt_tracks()
    class_name = gt_data.export_train_class()
    evaluateNfold(nfold=5,
                  dataset=df_fg_data,
                  class_name=class_name,
                  config=config_data,
                  groundTruth=None,
                  trainingFunc=None,
                  seed=None)
