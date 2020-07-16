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
from datetime import datetime

import logging
import pandas as pd
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from classification.gaia_imitation_best_model import evaluate_gaia_imitation_model
from transformation.load_groung_truth import ListGroundTruthFiles, GroundTruthLoad

from folding import export_folded_instances
from classification.classifierGRID import TrainGridClassifier
from classification.data_processing import DataProcessing


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

    if config["gaia_imitation"] is True:
        print("Gaia evaluation imitation for the best model of {} class is turned ON.".format(class_name))
        evaluate_gaia_imitation_model(config=config, class_name=class_name, X=X, y=y)
    elif config["gaia_imitation"] is False:
        print("Gaia evaluation imitation for the best model of {} class is turned OFF.".format(class_name))
    else:
        print("Please provide a correct boolean value for imitating or not gaia's best model for the {} class"
              .format(class_name))
    print()
    print()
    X, y = DataProcessing(config=config,
                          dataset=dataset,
                          class_name=class_name
                          ).exporting_classification_data()
    print()
    # TODO if not gaia imitation, train the model
    training_processes = TrainingProcesses(config).training_processes()
    print()
    exports_dir = "{}_{}".format(config.get("exports_directory"), class_name)
    exports_path = FindCreateDirectory(exports_dir).inspect_directory()
    print(exports_path)
    grid_svm_train = TrainGridClassifier(config=config,
                                         class_name=class_name,
                                         X=X,
                                         y=y,
                                         tr_processes=training_processes,
                                         n_fold=n_fold,
                                         exports_path=exports_path
                                         )
    grid_svm_train.export_best_classifier()

    print("Last evaluation took place at: {}".format(datetime.now()))


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
