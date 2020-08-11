import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
import yaml
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from transformation.transform import Transform
from classification.report_files_export import export_report
from logging_tool import LoggerSetup


def fold_evaluation(config, n_fold, X, y, class_name, tracks, process, exports_path, log_level):
    print(colored("------ EVALUATION and FOLDING ------", "yellow"))
    logger = LoggerSetup(config=config,
                         exports_path=exports_path,
                         name="evaluation_{}".format(class_name),
                         train_class=class_name,
                         mode="w",
                         level=log_level).setup_logger()

    logger.info("---- Folded evaluation of the model in the dataset ----")
    logger.info("number of folds set to config: {}".format(n_fold))
    logger.debug("Sample of shuffled tracks tracks:")
    logger.debug("{}".format(tracks[:5]))
    logger.debug("Tracks list length: {}".format(len(tracks)))
    # load project directory
    exports_dir = config.get("exports_directory")

    # load best model
    load_model_params_path = os.path.join(exports_path, exports_dir, "best_model_{}.json".format(class_name))
    with open(load_model_params_path) as model_params_file:
        model_params_data = json.load(model_params_file)

    logger.info("Best model preprocessing step: {}".format(process))
    models_path = FindCreateDirectory(exports_path,
                                      os.path.join(exports_dir, "models")).inspect_directory()
    # load the saved classifier
    clf = joblib.load(os.path.join(models_path, "model_grid_{}.pkl".format(process)))
    logger.info("Best model loaded.")

    # inner K-Fold cross-validation declaration
    random_seed = None
    shuffle = config["k_fold_shuffle"]
    if shuffle is True:
        random_seed = config["seed"]
    elif shuffle is False:
        random_seed = None
    logger.info("Fitting the data to the classifier with K-Fold cross-validation..")
    inner_cv = KFold(n_splits=n_fold,
                     shuffle=shuffle,
                     random_state=random_seed
                     )

    tracks_fold_indexing_dict = {}
    # transformation of the data to proper features based on the preprocess step
    features_prepared = Transform(config=config,
                                  df_feats=X,
                                  process=process,
                                  train_class=class_name,
                                  exports_path=exports_path,
                                  log_level=log_level).post_processing()
    logger.debug("Features prepared shape: {}".format(features_prepared.shape))

    # Starting Training, Evaluation for each fold
    logger.info("Starting fold-evaluation..")
    accuracy_model = []
    predictions_df_list = []
    fold_number = 0
    for train_index, test_index in inner_cv.split(features_prepared):
        logger.info("FOLD {} - Analyzing, Fitting, Predicting".format(fold_number))
        logger.debug("first test index element: {} - last test index element: {}".format(test_index[0], test_index[-1]))
        logger.debug("TEST INDEX: {}".format(test_index))
        logger.debug("Length of the test index array: {}".format(len(test_index)))

        # tracks indexing list for each fold
        tracks_count = 0
        tracks_list = []
        for index in test_index:
            tracks_fold_indexing_dict[tracks[index]] = fold_number
            tracks_list.append(tracks[index])
            tracks_count += 1
        logger.debug("Tracks indexed to the specific fold: {}".format(tracks_count))
        X_train, X_test = features_prepared[train_index], features_prepared[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model
        clf.fit(X_train, y_train)
        logger.debug("Classifier classes: {}".format(clf.classes_))
        # predictions for the features test
        pred = clf.predict(X_test)
        # predictions numpy array transformation to pandas DF
        df_pred = pd.DataFrame(data=pred, index=test_index, columns=["predictions"])
        # predictions' probabilities
        pred_prob = clf.predict_proba(X_test)
        # predictions' probabilities numpy array transformation to pandas DF
        df_pred_prob = pd.DataFrame(data=pred_prob, index=test_index, columns=clf.classes_)
        # tracks list transformation to pandas DF
        df_tracks = pd.DataFrame(data=tracks_list, index=test_index, columns=["track"])
        logger.debug("\n{}".format(df_tracks.head()))
        # y_test pandas Series transformation to pandas DF
        y_test_series = pd.DataFrame(data=y_test, index=test_index, columns=[class_name])
        # concatenate the 4 DFs above to 1 for saving the resulted dataset
        # (tracks, predictions' probabilities, predictions, true)
        logger.debug("Concatenating DF..")
        df_pred_general = pd.concat([df_tracks, df_pred_prob, df_pred, y_test_series], axis=1, ignore_index=False)
        # Append the folded dataset to a list that will contain all the folded datasets:
        predictions_df_list.append(df_pred_general)
        # Append each accuracy of the folded model to a list that contains all the accuracies resulted from each fold
        accuracy_model.append(accuracy_score(y_test, clf.predict(X_test), normalize=True) * 100)
        fold_number += 1
    # concatenate the folded predictions DFs
    logger.info("Make Predictions DataFrame for all the folded instances together..")
    df_predictions = pd.concat(predictions_df_list)
    logger.debug("\n{}".format(df_predictions.head()))
    logger.debug("Info:")
    logger.debug("\n{}".format(df_predictions.info()))
    # save predictions df
    logger.info("Saving the unified predictions DataFrame locally.")
    dataset_path = FindCreateDirectory(exports_path,
                                       os.path.join(exports_dir, "dataset")).inspect_directory()
    df_predictions.to_csv(os.path.join(dataset_path, "predictions_{}.csv".format(class_name)))
    print()
    # ACCURACIES in each fold
    logger.info("Accuracies in each fold: {}".format(accuracy_model))
    logger.info("Mean of accuracies: {}".format(np.mean(accuracy_model)))
    logger.info("Standard Deviation of accuracies: {}".format(np.std(accuracy_model)))
    accuracies_export = "Accuracies in each fold: {} \nMean of accuracies: {} \nStandard Deviation of accuracies: {}"\
        .format(accuracy_model, np.mean(accuracy_model), np.std(accuracy_model))
    export_report(config=config,
                  name="Accuracies results",
                  report=accuracies_export,
                  filename="accuracies_results_fold",
                  train_class=class_name,
                  exports_path=exports_path)

    # Visualize accuracy for each iteration
    logger.info("Visualize accuracy for each iteration..")
    list_folds = []
    counter_folds = 0
    for accuracy in accuracy_model:
        list_folds.append("Fold{}".format(counter_folds))
        counter_folds += 1
    logger.debug("Exporting accuracies distribution to plot file..")
    scores = pd.DataFrame(accuracy_model, columns=['Scores'])
    sns.set(style="white", rc={"lines.linewidth": 3})
    sns.barplot(x=list_folds, y="Scores", data=scores)
    images_path = FindCreateDirectory(exports_path,
                                      os.path.join(exports_dir, "images")).inspect_directory()
    plt.savefig(os.path.join(images_path, "accuracies_distribution.png"))
    sns.set()
    plt.close()
    logger.info("Plot saved successfully.")

    # Folded Tracks Dictionary
    logger.info("Writing Folded Tracks Dictionary locally to check where each track is folded..")
    logger.debug("length of keys: {}".format(len(tracks_fold_indexing_dict.keys())))
    folded_dataset_path = os.path.join(dataset_path,  "{}.yaml".format(class_name))
    with open(folded_dataset_path, 'w') as file:
        folded_dataset = yaml.dump(tracks_fold_indexing_dict, file)
    logger.info("Folded dataset written successfully to disk.")

    # ------------ Evaluation to the folded Dataset ------------
    logger.info("---- Evaluation to the folded dataset ----")
    # Confusion Matrix
    logger.info("Exporting Confusion Matrix applied to the folded dataset..")
    cm = confusion_matrix(y_true=df_predictions[class_name], y_pred=df_predictions["predictions"])
    logger.info("\n{}".format(cm))
    # Confusion Matrix Normalized
    logger.info("Exporting Normalized Confusion Matrix applied to the folded dataset..")
    cm_normalized = (cm / cm.astype(np.float).sum(axis=1) * 100)
    logger.info("\n{}".format(cm_normalized))
    cm_all = "Actual instances\n{}\n\nNormalized\n{}".format(cm, cm_normalized)
    # export the confusion matrix report for the folded dataset
    export_report(config=config,
                  name="Folded Data Confusion Matrix",
                  report=cm_all,
                  filename="confusion_matrix_fold",
                  train_class=class_name,
                  exports_path=exports_path)
    # Classification Report
    logger.info("Exporting Classification Report applied to the folded dataset..")
    cr = classification_report(y_true=df_predictions[class_name], y_pred=df_predictions["predictions"])
    # export the Classification report for the whole dataset
    export_report(config=config,
                  name="Folded Data Classification Report",
                  report=cr,
                  filename="classification_report_fold",
                  train_class=class_name,
                  exports_path=exports_path)
    logger.info("The folded dataset has been evaluated successfully..")
    print()
    # ------------ Training and Evaluation to the whole Dataset ------------
    print(colored("Evaluation to the whole dataset..", "cyan"))
    logger.info("---- Evaluation to the whole dataset ----")
    # train to the whole dataset
    clf.fit(features_prepared, y)
    # prediction for the whole dataset
    predictions_all = clf.predict(features_prepared)
    logger.info("Confusion Matrix applied to the whole dataset..")
    cm_full = confusion_matrix(y_true=y, y_pred=predictions_all)
    logger.info("\n{}".format(cm_full))
    logger.info("Normalized Confusion Matrix applied to the whole dataset..")
    cm_full_normalized = (cm_full / cm_full.astype(np.float).sum(axis=1) * 100)
    logger.info("\n{}".format(cm_full_normalized))
    cm_full_all = "Actual instances\n{}\n\nNormalized\n{}".format(cm_full, cm_full_normalized)
    # export the confusion matrix report for the whole dataset
    export_report(config=config,
                  name="All Data Confusion Matrix",
                  report=cm_full_all,
                  filename="confusion_matrix_all_dataset",
                  train_class=class_name,
                  exports_path=exports_path)
    logger.info("Classification Report applied to the whole dataset..")
    cr_full = classification_report(y_true=y, y_pred=predictions_all)
    # export the Classification report for the whole dataset
    export_report(config=config,
                  name="All Data Classification Report",
                  report=cr_full,
                  filename="classification_report_all_dataset",
                  train_class=class_name,
                  exports_path=exports_path)
    logger.info("The whole dataset has been evaluated successfully..")

