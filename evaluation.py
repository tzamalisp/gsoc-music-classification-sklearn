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
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from gaia_imitation_best_model import evaluate_gaia_imitation_model
from ml_load_groung_truth import ListGroundTruthFiles, GroundTruthLoad, DatasetDFCreator
from transformation.features_labels import FeaturesLabelsSplitter
from folding import export_folded_instances
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from transformation.transform import Transform
from sklearn.model_selection import KFold

def evaluate(classifier, dataset, groundTruth, confusion=None, nfold=None, verbose=True):
    """Evaluate the classifier on the given dataset and returns the confusion matrix.

    Uses only the points that are in the groundTruth parameter for the evaluation.

    Parameters
    ----------

    classifier  : a function which given a point returns its class
    dataset     : the dataset from which to get the points
    groundTruth : a map from the points to classify to their respective class
    """

    # progress = TextProgress(len(groundTruth))
    # done = 0
    #
    # confusion = confusion or ConfusionMatrix()
    #
    # for pointId, expected in groundTruth.items():
    #     try:
    #         found = classifier(dataset.point(pointId))
    #         if nfold is None:
    #             confusion.add(expected, found, pointId)
    #         else:
    #             confusion.addNfold(expected, found, pointId, nfold)
    #
    #     except Exception as e:
    #         log.warning('Could not classify point "%s" because %s' % (pointId, str(e)))
    #         raise
    #
    #     done += 1
    #     if verbose: progress.update(done)
    #
    # return confusion





def evaluateNfold(n_fold, dataset, groundTruth, trainingFunc, config, seed=None, *args, **kwargs):
    """Evaluate the classifier on the given dataset and returns the confusion matrix.

    The evaluation is performed using n-fold cross validation.
    Uses only the points that are in the groundTruth parameter for the evaluation.

    Parameters
    ----------

    n_fold        : the number of folds to use for the cross-validation
    dataset      : the dataset from which to get the points
    groundTruth  : a map from the points to classify to their respective class
    trainingFunc : a function which will train and return a classifier given a dataset,
                   the groundtruth, and the *args and **kwargs arguments
    """
    n_fold = config["gaia_kfold_cv_n_splits"]
    log.info('Doing %d-fold cross validation' % n_fold)

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
        print("Gaia evaluation imitation for the best model of {} class is turned ON.".format(class_name))
        evaluate_gaia_imitation_model(config=config, class_name=class_name, X=X, y=y)
    elif config["gaia_imitation"] is False:
        print("Gaia evaluation imitation for the best model of {} class is turned OFF.".format(class_name))
    else:
        print("Please provide a correct boolean value for imitating or not gaia's best model for the {} class"
              .format(class_name))

    tr_processes = TrainingProcesses(config).training_processes()
    process_counter = 0
    for tr_process in tr_processes:
        print("Train process {} - {}".format(process_counter, tr_process))
        # evaluate()

        if tr_process["classifier"]:
            exports_dir = "{}_{}".format(config.get("exports_directory"), class_name)
            exports_path = FindCreateDirectory(exports_dir).inspect_directory()
            print(exports_path)
            X_transformed = Transform(config=config,
                                      df=X,
                                      process=tr_process["preprocess"],
                                      exports_path=exports_path
                                      ).post_processing()

            # define the length of parameters
            parameters_grid = {'kernel': tr_process["kernel"],
                               'C': tr_process["C"],
                               'gamma': tr_process["gamma"],
                               'class_weight': tr_process["balanceClasses"]
                               }

            svm = SVC(probability=True)

            # To be used within GridSearch without gaia imitation
            inner_cv = KFold(n_splits=n_fold,
                             shuffle=False,
                             )

            gsvc = GridSearchCV(estimator=svm,
                                param_grid=parameters_grid,
                                cv=inner_cv,
                                n_jobs=config["parallel_jobs"])

            print("Shape of X before train: {}".format(X_transformed.shape))
            print("Fitting the data to the model:")
            gsvc.fit(X_transformed, y)

            # print(gsvc.cv_results_["params"])
            print(gsvc.best_score_)
            print(gsvc.best_estimator_)
            print(gsvc.best_params_)
            print("Counted evaluations in this GridSearch Train process: {}".format(len(gsvc.cv_results_["params"])))

        print()
        print("Next train process..")
        print()
        print()
        process_counter += 1


if __name__ == '__main__':
    log = logging.getLogger('classification.Evaluation')

    config_data = load_yaml("configuration.yaml")

    gt_data = GroundTruthLoad(config_data, "groundtruth.yaml")
    df_fg_data = gt_data.export_gt_tracks()
    class_name = gt_data.export_train_class()
    evaluateNfold(n_fold=5,
                  dataset=df_fg_data,
                  class_name=class_name,
                  config=config_data,
                  groundTruth=None,
                  trainingFunc=None,
                  seed=None)
