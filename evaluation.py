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
from utils import load_yaml
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from ml_load_groung_truth import ListGroundTruthFiles, GroundTruthLoad


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
    log.info('Doing %d-fold cross validation' % nfold)
    # classes = set(groundTruth.values())
    # progress = TextProgress(nfold, 'Evaluating fold %(current)d/%(total)d')

    # get map from class to point names
    # iclasses = {}
    # for c in classes:
    #     iclasses[c] = [ p for p in groundTruth.keys() if groundTruth[p] == c ]
    #     random.seed(a=seed)
    #     random.shuffle(iclasses[c])
    dataset_merge = dataset.copy()
    dataset_merge['folder_name'] = dataset_merge[["json_directory", "track"]].apply(lambda x: ''.join(x), axis=1)

    dataset_merge = dataset_merge.drop(columns=["json_directory", "track", "track_path", "danceability"], axis=1)
    print(dataset_merge.tail(20))
    print()
    print("Find if indexing is ok:")
    print(dataset_merge["folder_name"].iloc[0])
    print(dataset_merge["folder_name"].iloc[4])
    print()
    print("Find duplicated rows:")
    duplicateRowsDF = dataset_merge[dataset_merge.duplicated()]
    print(duplicateRowsDF)
    print()

    print("Transform to numpy array:")
    X_array = dataset_merge.values
    X_array_list = X_array.tolist()
    print(X_array[0])
    print(X_array[4])

    print("Shuffle the data:")
    random.seed(a=config.get("random_seed"))
    random.shuffle(X_array_list)
    print("Check some indexes:")
    pprint(X_array_list[:10])
    print("Shuffle array length: {}".format(len(X_array)))

    print("Folding..")
    # Train the classifier with K-Fold cross-validation
    random_seed = None
    # shuffle = config["k_fold_shuffle"]
    shuffle = False
    if shuffle is True:
        random_seed = config["k_fold_shuffle"]
    elif shuffle is False:
        random_seed = None
    print("Fitting the data to the classifier with K-Fold cross-validation..")
    kf = KFold(n_splits=config["gaia_kfold_cv_n_splits"],
               shuffle=shuffle,
               random_state=random_seed
               )
    print()
    print()
    # tracks_fold_indexing = []
    tracks_fold_indexing_dict = {}
    tracks_fold_indexing_list = []
    print(X_array_list[0])
    print(X_array_list[4])
    fold_number = 0
    for train_index, val_index in kf.split(X_array_list):

        print("Fold: {}".format(fold_number))
        # print("TRAIN INDEX: ", train_index)
        print("TEST INDEX: ", val_index)
        # print(len(train_index))
        print("Length of the test index array: {}".format(len(val_index)))

        tracks_count = 0
        for index in val_index:
            # track = df_shuffled["folder_name"].iloc[index]
            track = X_array_list[index][0]
            # print(track)
            tracks_fold_indexing_dict[track] = fold_number
            tracks_fold_indexing_list.append("{}: {}".format(track, fold_number))
            tracks_count += 1
        print("tracks counted:", tracks_count)
        fold_number += 1
    print()
    print()
    print(["Dictionary:"])
    pprint(tracks_fold_indexing_dict)
    print("length of keys:", len(tracks_fold_indexing_dict.keys()))
    print()
    print()



    # # get folds
    # folds = {}
    # for i in range(nfold):
    #     folds[i] = []
    #     for c in iclasses.values():
    #         foldsize = (len(c)-1)//nfold + 1 # -1/+1 so we take all instances into account, last fold might have fewer instances
    #         folds[i] += c[ foldsize * i : foldsize * (i+1) ]
    #
    # # build sub-datasets and run evaluation on them
    # confusion = None
    # pnames = [ p.name() for p in dataset.points() ]
    #
    # for i in range(nfold):
    #     if log.isEnabledFor(logging.INFO):
    #         progress.update(i+1)
    #
    #     trainds = DataSet()
    #     trainds.addPoints([ dataset.point(pname) for pname in pnames if pname not in folds[i] ])
    #     traingt = GroundTruth(groundTruth.className, dict([ (p, c) for p, c in groundTruth.items() if p not in folds[i] ]))
    #
    #     testds = DataSet()
    #     testds.addPoints([ dataset.point(str(pname)) for pname in folds[i] ])
    #     testgt = GroundTruth(groundTruth.className, dict([ (p, c) for p, c in groundTruth.items() if p in folds[i] ]))
    #
    #     classifier = trainingFunc(trainds, traingt, *args, **kwargs)
    #     # confusion = evaluate(classifier, testds, testgt, confusion, nfold=i, verbose=False)
    #
    # return confusion
    # return df_shuffled


if __name__ == '__main__':
    log = logging.getLogger('classification.Evaluation')

    config_data = load_yaml("configuration.yaml")

    gt_data = GroundTruthLoad(config_data, "groundtruth.yaml")
    df_fg_data = gt_data.create_df_tracks()
    class_name = gt_data.export_class_name()
    evaluateNfold(5, df_fg_data, groundTruth=None, trainingFunc=None, seed=None, config=config_data)

