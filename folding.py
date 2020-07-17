import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pprint import pprint
import yaml
from transformation.transform import Transform


def export_folded_instances(config, clf, n_fold, X_array_list, y, class_name, tracks, process, exports_path):
    print(colored("Folding..", "yellow"))
    print(colored("n_fold: {}".format(n_fold), "cyan"))
    print(colored("Sample of shuffled tracks tracks:", "cyan"))
    pprint(tracks[:5])
    # Train the classifier with K-Fold cross-validation
    random_seed = None
    # shuffle = config["k_fold_shuffle"]
    shuffle_f_fold = False
    if shuffle_f_fold is True:
        random_seed = config["k_fold_random_seed"]
    elif shuffle_f_fold is False:
        random_seed = None
    print("Fitting the data to the classifier with K-Fold cross-validation..")
    kf = KFold(n_splits=n_fold,
               shuffle=shuffle_f_fold,
               random_state=random_seed
               )
    print()
    print()
    print(colored("Type of X: {}".format(type(X_array_list)), "cyan"))
    print(colored("Type of y: {}".format(type(y)), "cyan"))
    # tracks_fold_indexing = []
    tracks_fold_indexing_dict = {}
    tracks_fold_indexing_list = []
    print(tracks[0])
    print(tracks[4])

    # transformation of the data
    X_transformed = Transform(config=config,
                              df=X_array_list,
                              process=process,
                              exports_path=exports_path
                              ).post_processing()

    accuracy_model = []
    predictions_list = pd.DataFrame()
    fold_number = 0
    for train_index, test_index in kf.split(X_transformed):
        print("Fold: {}".format(fold_number))
        # print("TRAIN INDEX: ", train_index)
        print("first test index element: {} - last test index element: {}".format(test_index[0], test_index[-1]))
        # print("TEST INDEX: ", test_index)
        print(colored("Length of the train index array: {}".format(len(train_index)), "cyan"))
        print(colored("Length of the test index array: {}".format(len(test_index)), "cyan"))

        tracks_count = 0
        for index in test_index:
            # track = df_shuffled["folder_name"].iloc[index]
            track = tracks[index][0]
            # print(track)
            tracks_fold_indexing_dict[track] = fold_number

            tracks_fold_indexing_list.append("{}: {}".format(track, fold_number))
            tracks_count += 1
        print(colored("Tracks indexed to the specific fold: {}".format(tracks_count), "cyan"))

        X_train, X_test = X_transformed.iloc[train_index], X_transformed.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train the model
        clf.fit(X_train, y_train)

        predictions_fold = clf.predict(X_test)
        print(type(predictions_fold))
        print(predictions_fold.shape)

        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, clf.predict(X_test), normalize=True) * 100)

        fold_number += 1
    # print(predictions_list)

    print()
    # ACCURACIES
    print(colored("Accuracies in each fold: {}".format(accuracy_model), "cyan"))
    print(colored("Mean of accuracies: {}".format(np.mean(accuracy_model)), "cyan"))
    print(colored("Standard Deviation of accuracies: {}".format(np.std(accuracy_model)), "cyan"))

    # Visualize accuracy for each iteration
    list_folds = []
    counter_folds = 0
    for accuracy in accuracy_model:
        list_folds.append("Fold{}".format(counter_folds))
        counter_folds += 1
    print("Exporting accuracis distribution to plot file..")
    scores = pd.DataFrame(accuracy_model, columns=['Scores'])
    sns.set(style="white", rc={"lines.linewidth": 3})
    sns.barplot(x=list_folds, y="Scores", data=scores)
    plt.savefig(os.path.join(exports_path, "images", "accuracies_distribution.png"))
    sns.set()
    plt.close()
    print("Plot saved successfully.")

    #
    print()
    print()
    # print("Dictionary:")
    # pprint(tracks_fold_indexing_dict)
    print("length of keys:", len(tracks_fold_indexing_dict.keys()))
    print()
    print()
    dataset_path = os.path.join(exports_path, "dataset",  "{}.yaml".format(class_name))
    with open(dataset_path, 'w') as file:
        folded_dataset = yaml.dump(tracks_fold_indexing_dict, file)
    print(colored("Folded dataset written successfully.", "cyan"))
