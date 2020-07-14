import os
from sklearn.model_selection import KFold
from pprint import pprint
import yaml


def export_folded_instances(config, n_fold, X_array_list, class_name, tr_process_number):
    print("Folding..")
    # Train the classifier with K-Fold cross-validation
    random_seed = None
    # shuffle = config["k_fold_shuffle"]
    shuffle_f_fold = False
    if shuffle_f_fold is True:
        random_seed = config["k_fold_shuffle"]
    elif shuffle_f_fold is False:
        random_seed = None
    print("Fitting the data to the classifier with K-Fold cross-validation..")
    kf = KFold(n_splits=n_fold,
               shuffle=shuffle_f_fold,
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
        print("tracks indexed to the specific fold:", tracks_count)
        fold_number += 1

    print()
    print()
    print("Dictionary:")
    pprint(tracks_fold_indexing_dict)
    print("length of keys:", len(tracks_fold_indexing_dict.keys()))
    print()
    print()
    export_dir = "{}_{}".format(config["exports_directory"], class_name)
    export_path = os.path.join(os.getcwd(), export_dir, "datasets", "{}.yaml")
    with open(export_path, 'w') as file:
        folded_dataset = yaml.dump(tracks_fold_indexing_dict, file)
