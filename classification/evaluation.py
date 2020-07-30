import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from pprint import pprint
import yaml
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import requests
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from transformation.transform import Transform
from transformation.utils_preprocessing import flatten_dict_full
from classification.report_files_export import export_report


def fold_evaluation(config, n_fold, X, y, class_name, tracks, process, exports_path):
    print(colored("Folding..", "yellow"))
    print(colored("n_fold: {}".format(n_fold), "cyan"))
    print(colored("Sample of shuffled tracks tracks:", "cyan"))
    pprint(tracks[:5])
    print("Tracks list length", len(tracks))

    exports_dir = "{}_{}".format(config.get("exports_directory"), class_name)

    # load best model
    load_model_params_path = os.path.join(exports_path, exports_dir, "best_model_{}.json".format(class_name))
    with open(load_model_params_path) as model_params_file:
        model_params_data = json.load(model_params_file)
    
    print("Best model preprocessing step: {}".format(process))
    models_path = FindCreateDirectory(exports_path,
                                      os.path.join(exports_dir, "models")).inspect_directory()
    clf = joblib.load(os.path.join(models_path, "model_grid_{}.pkl".format(process)))
    print("Best model loaded.")

    # inner with K-Fold cross-validation declaration
    random_seed = None
    shuffle = config["k_fold_shuffle"]
    if shuffle is True:
        random_seed = config["seed"]
    elif shuffle is False:
        random_seed = None
    print("Fitting the data to the classifier with K-Fold cross-validation..")
    inner_cv = KFold(n_splits=n_fold,
                     shuffle=shuffle,
                     random_state=random_seed
                     )
    print()
    print()
    print(colored("Type of X: {}".format(type(X)), "cyan"))
    print(colored("Type of y: {}".format(type(y)), "cyan"))
    # tracks_fold_indexing = []
    tracks_fold_indexing_dict = {}
    print(tracks[0])
    print(tracks[4])

    # transformation of the data
    features_prepared = Transform(config=config,
                                  df_feats=X,
                                  process=process,
                                  train_class=class_name,
                                  exports_path=exports_path).post_processing()
    print("features prepared shape: {}".format(features_prepared.shape))

    accuracy_model = []
    predictions_df_list = []
    fold_number = 0
    for train_index, test_index in inner_cv.split(features_prepared):
        print("Fold: {}".format(fold_number))
        # print("TRAIN INDEX: ", train_index)
        print("first test index element: {} - last test index element: {}".format(test_index[0], test_index[-1]))
        print("TEST INDEX: ", test_index)
        print(colored("Length of the train index array: {}".format(len(train_index)), "cyan"))
        print(colored("Length of the test index array: {}".format(len(test_index)), "cyan"))

        tracks_count = 0
        tracks_list = []
        for index in test_index:
            # print(tracks[index])
            tracks_fold_indexing_dict[tracks[index]] = fold_number
            tracks_list.append(tracks[index])
            tracks_count += 1
        print(colored("Tracks indexed to the specific fold: {}".format(tracks_count), "cyan"))
        print(type(tracks_list))
        X_train, X_test = features_prepared[train_index], features_prepared[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model
        clf.fit(X_train, y_train)
        print("Classifier classes: {}".format(clf.classes_))
        # predictions
        print("Predictions:")
        pred = clf.predict(X_test)
        print(type(pred))
        df_pred = pd.DataFrame(data=pred, index=test_index, columns=["predictions"])
        print(type(pred))
        print(pred.shape)
        print(df_pred.head())
        # predictions probabilities
        print("Predictions Probabilities:")
        pred_prob = clf.predict_proba(X_test)
        df_pred_prob = pd.DataFrame(data=pred_prob, index=test_index, columns=clf.classes_)
        print(df_pred_prob.head())
        print("Tracks:")
        # tracks df
        df_tracks = pd.DataFrame(data=tracks_list, index=test_index, columns=["track"])
        print(df_tracks.head())
        # y_test series
        print("True values:")
        y_test_series = pd.DataFrame(data=y_test, index=test_index, columns=[class_name])
        print(y_test_series.head())
        # concatenate dfs
        df_pred_general = pd.concat([df_tracks, df_pred_prob, df_pred, y_test_series], axis=1, ignore_index=False)
        print(df_pred_general.head())
        # predictions_all_df.append(df_pred_general, ignore_index=True)
        predictions_df_list.append(df_pred_general)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, clf.predict(X_test), normalize=True) * 100)
        fold_number += 1

    print()
    print()
    # concatenate predictions dfs
    print(colored("DF Predictions all:", "cyan"))
    df_predictions = pd.concat(predictions_df_list)
    print(df_predictions.head())
    print("Info:")
    print(df_predictions.info())
    # save predictions df
    dataset_path = FindCreateDirectory(exports_path,
                                       os.path.join(exports_dir, "dataset")).inspect_directory()
    df_predictions.to_csv(os.path.join(dataset_path, "predictions_{}.csv".format(class_name)))
    print()
    # ACCURACIES
    print(colored("Accuracies in each fold: {}".format(accuracy_model), "cyan"))
    print(colored("Mean of accuracies: {}".format(np.mean(accuracy_model)), "cyan"))
    print(colored("Standard Deviation of accuracies: {}".format(np.std(accuracy_model)), "cyan"))
    accuracies_export = "Accuracies in each fold: {} \nMean of accuracies: {} \nStandard Deviation of accuracies: {}"\
        .format(accuracy_model, np.mean(accuracy_model), np.std(accuracy_model))
    export_report(config=config,
                  name="Accuracies results",
                  report=accuracies_export,
                  filename="accuracies_results_fold",
                  train_class=class_name,
                  exports_path=exports_path)

    # Visualize accuracy for each iteration
    list_folds = []
    counter_folds = 0
    for accuracy in accuracy_model:
        list_folds.append("Fold{}".format(counter_folds))
        counter_folds += 1
    print("Exporting accuracies distribution to plot file..")
    scores = pd.DataFrame(accuracy_model, columns=['Scores'])
    sns.set(style="white", rc={"lines.linewidth": 3})
    sns.barplot(x=list_folds, y="Scores", data=scores)
    images_path = FindCreateDirectory(exports_path,
                                      os.path.join(exports_dir, "images")).inspect_directory()
    plt.savefig(os.path.join(images_path, "accuracies_distribution.png"))
    sns.set()
    plt.close()
    print("Plot saved successfully.")

    # Folded Tracks Dictionary
    print(colored("Folded Tracks Dictionary"))
    # print("Dictionary:")
    # pprint(tracks_fold_indexing_dict)
    print("length of keys:", len(tracks_fold_indexing_dict.keys()))
    print("Saving folded dataset..")
    folded_dataset_path = os.path.join(dataset_path,  "{}.yaml".format(class_name))
    with open(folded_dataset_path, 'w') as file:
        folded_dataset = yaml.dump(tracks_fold_indexing_dict, file)
    print(colored("Folded dataset written successfully to disk.", "cyan"))

    print()
    print(colored("Evaluation Reports:", "cyan"))
    # CONFUSION MATRIX
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true=df_predictions[class_name], y_pred=df_predictions["predictions"])
    cm_normalized = (cm / cm.astype(np.float).sum(axis=1) * 100)
    print(colored("Regular:", "green"))
    print(colored(cm, "green"))
    print(colored("Normalized:", "yellow"))
    # print(colored(cm_normalized, "yellow"))
    cm_all = "Actual instances\n{}\n\nNormalized\n{}".format(cm, cm_normalized)
    export_report(config=config,
                  name="Confusion Matrix",
                  report=cm_all,
                  filename="confusion_matrix_fold",
                  train_class=class_name,
                  exports_path=exports_path)

    print()
    print("Classification Report:")
    cr = classification_report(y_true=df_predictions[class_name], y_pred=df_predictions["predictions"])
    print(cr)
    export_report(config=config,
                  name="Classification Report",
                  report=cr,
                  filename="classification_report_fold",
                  train_class=class_name,
                  exports_path=exports_path)

    # # save the model
    # models_path = FindCreateDirectory(os.path.join(exports_path, "models")).inspect_directory()
    # model_save_path = os.path.join(models_path, "model.pkl")
    # joblib.dump(clf, model_save_path)
    #
    # train with all the data of the dataset
    print(colored("Evaluation to the whole dataset..", "cyan"))
    clf.fit(features_prepared, y)
    predictions_proba_all = clf.predict_proba(features_prepared)
    predictions_all = clf.predict(features_prepared)
    print(colored("Confusion Matrix All:", "magenta"))
    cm_full = confusion_matrix(y_true=y, y_pred=predictions_all)
    print(cm_full)
    print(colored("Confusion Matrix All Normalized:", "magenta"))
    cm_full_normalized = (cm_full / cm_full.astype(np.float).sum(axis=1) * 100)
    print(cm_full_normalized)
    cm_full_all = "Actual instances\n{}\n\nNormalized\n{}".format(cm_full, cm_full_normalized)
    export_report(config=config,
                  name="Confusion Matrix",
                  report=cm_full_all,
                  filename="confusion_matrix_all_dataset",
                  train_class=class_name,
                  exports_path=exports_path)
    print(colored("Classification Report All:", "magenta"))
    cr_full = classification_report(y_true=y, y_pred=predictions_all)
    print(cr_full)
    export_report(config=config,
                  name="Classification Report",
                  report=cr_full,
                  filename="classification_report_all_dataset",
                  train_class=class_name,
                  exports_path=exports_path)

