import os
import pandas as pd
import numpy as np
from utils import load_yaml, FindCreateDirectory
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import yaml
# Neural Networks
import tensorflow as tf
from tensorflow import keras
import time


def display_scores(scores):
    print("Display scores:")
    print("Scores: {}".format(scores))
    print("Mean: {}".format(scores.mean()))
    print("Standard Deviation: {}".format(scores.std()))


class Models:
    """
    Todo: Commenting
    """
    def __init__(self, config, features, labels, class_name, exports_path, tracks_shuffled):
        """

        :param config:
        :param features:
        :param labels:
        :param class_name:
        """
        self.config = config
        self.features = features
        self.labels = labels
        self.class_name = class_name
        self.exports_path = exports_path
        self.tracks_shuffled = tracks_shuffled
        # self.df_fold = pd.DataFrame()

    def train_svm(self):
        """

        :return:
        """


        # list processes
        svm_processes = self.config["classifiers"]["svm"][0]
        list_processes = []
        for classifier in self.config["classifiers"].keys():
            for ker in svm_processes["kernel"]:
                for c in svm_processes["C"]:
                    for balance in svm_processes["balanceClasses"]:
                        for preprocess in svm_processes["preprocessing"]:
                            for typeness in svm_processes["type"]:
                                for gamma in svm_processes["gamma"]:
                                    dict_process = {}
                                    dict_process["kernel"] = ker
                                    dict_process["C"] = c
                                    dict_process["balanceClasses"] = balance
                                    dict_process["preprocessing"] = preprocess
                                    dict_process["type"] = typeness
                                    dict_process["classifier"] = classifier
                                    dict_process["gamma"] = gamma
                                    list_processes.append(dict_process)
        print("Number of processes to be executed: {}".format(len(list_processes)))
        list_evaluations = []
        for evaluation in self.config["evaluations"].keys():
            for test in self.config["evaluations"][evaluation]:
                for key in test:
                    for value in test[key]:
                        dict_eval = {}
                        dict_eval["nfold"] = value
                        dict_eval["type"] = evaluation
                        list_evaluations.append(dict_eval)
        print("Number of evaluations to be executed: {}".format(len(list_evaluations)))

        # read gaia params
        path_script = os.getcwd()
        path_app = os.path.join(path_script)
        with open(os.path.join(path_app, "gaia_best_models", "jmp_results_{}.param".format(self.class_name)), "r") as f:
            gaia_model = yaml.safe_load(f)

        print(gaia_model)

        svc_c = 1.0
        svc_gamma = 'auto'
        # C = 2 ** C_value
        if self.config.get("svm_C") != "" and self.config.get("svm_C") is not None:
            svc_c = 2 ** self.config.get("svm_C")
        # gamma = 2 ** gamma_value
        if self.config.get("svc_gamma") != "" and self.config.get("svc_gamma") is not None:
            svc_gamma = 2 ** self.config.get("svc_gamma")

        # svm = SVC(
        #     C=svc_c,
        #     kernel=self.config.get("svc_kernel"),
        #     gamma=svc_gamma,
        #     class_weight=self.config.get("svc_class_weight_balance"),
        #     probability=self.config.get("svc_probability")
        # )

        gaia_C = 2 ** gaia_model["model"]["C"]
        gaia_kernel = gaia_model["model"]["kernel"].lower()
        gaia_gamma = 2 ** gaia_model["model"]["gamma"]
        gaia_class_weight = None
        if  gaia_model["model"]["balanceClasses"] is True:
            gaia_class_weight = "balanced"
        elif gaia_model["model"]["balanceClasses"] is False:
            gaia_class_weight = None

        svm = SVC(
            C=gaia_C,
            kernel=gaia_kernel,
            gamma=gaia_gamma,
            class_weight=gaia_class_weight,
            probability=self.config.get("svc_probability")
        )

        if self.config.get("k_fold_apply") is True:
            # KFold Cross Validation approach
            X = self.features
            print("Type of X (features):", type(X))
            y = self.labels
            print("Type of y (labels):", type(y))

            # transform DF to np array for K-Fold SVM training
            X_array = X.values
            print("Type of DF to array tranformation for F-fold training: {}".format(type(X_array)))
            print("Shape of this array transgormation: {}".format(X_array.shape))

            # Use cross-validation by calling the cross_val_score helper function on the estimator and the dataset.
            print()
            # Evaluate the classifier with cross_val_score
            exports_dir = os.path.join(self.exports_path, self.config.get("evaluations_directory"))
            exports_path = FindCreateDirectory(exports_dir).inspect_directory()

            print("Evaluate the classifier with cross_val_score:")
            scores = cross_val_score(estimator=svm,
                                     X=X_array,
                                     y=self.labels,
                                     scoring="accuracy",
                                     cv=self.config.get("k_fold"),
                                     n_jobs=self.config.get("parallel_jobs"),
                                     verbose=self.config.get("verbose")
                                     )
            print()
            print("Score results:")
            display_scores(scores)
            print()
            print()

            # Evaluate the classifier with cross_validate
            print("Evaluate the classifier with cross_validate:")
            cv_results = cross_validate(estimator=svm,
                                        X=X_array,
                                        y=self.labels,
                                        scoring="accuracy",
                                        cv=self.config.get("k_fold"),
                                        n_jobs=self.config.get("parallel_jobs"),
                                        verbose=self.config.get("verbose"),
                                        return_estimator=True
                                        )

            print("CV results:")
            print(sorted(cv_results.keys()))
            for item in cv_results.keys():
                print("{}: {}".format(item, cv_results[item]))
            print("Mean of all the accuracies: {}".format(sum(cv_results["test_score"]) / len(cv_results["test_score"])))
            print()

            # Evaluate the classifier with cross_val_predict
            print("Evaluate the classifier with cross_val_predict:")
            predictions = cross_val_predict(estimator=svm,
                                            X=X_array,
                                            y=self.labels,
                                            cv=self.config.get("k_fold"),
                                            n_jobs=self.config.get("parallel_jobs"),
                                            verbose=self.config.get("verbose")
                                            )

            print("Confusion matrix from cross_val_predict:")
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_true=self.labels, y_pred=predictions))
            # print(predictions.shape)
            print()
            print("Normalized Confusion matrix from cross_val_predict:")
            cm = confusion_matrix(y_true=self.labels, y_pred=predictions)
            cm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100
            print(cm)
            print()

            # Train the classifier with K-Fold cross-validation
            random_seed = None
            shuffle = self.config["k_fold_shuffle"]
            if shuffle is True:
                random_seed = self.config["k_fold_shuffle"]
            elif shuffle is False:
                random_seed = None
            print("Fitting the data to the classifier with K-Fold cross-validation..")
            kf = KFold(n_splits=self.config["gaia_kfold_cv_n_splits"],
                       shuffle=shuffle,
                       random_state=random_seed
                       )
            # Initialize the accuracy of the models to blank list.
            # The accuracy of each model will be appended to this list
            accuracy_model = []
            # Iterate over each train-test split
            for train_index, val_index in kf.split(X_array):
                print("TRAIN INDEX: ", train_index)
                print("TEST INDEX: ", val_index)
                # Split train-test
                X_train, X_val = X_array[train_index], X_array[val_index]
                y_train, y_val = y[train_index], y[val_index]
                df = self.tracks_shuffled[["json_directory", "track"]].iloc[val_index]
                # print(df)
                # print(df.columns)
                # Train the model
                model = svm.fit(X_train, y_train)
                predictions = svm.predict(X_val)
                predictions_df = pd.DataFrame(predictions, index=val_index, columns=["predictions"])
                # y_true_df = pd.DataFrame(y_val, index=val_index, columns=["true"])
                fold_concat_df = pd.concat([df, predictions_df, y_val], axis=1)
                print(fold_concat_df)
                print(fold_concat_df.columns)
                print()
                # Append to accuracy_model the accuracy of the model
                accuracy_model.append(accuracy_score(y_val, model.predict(X_val), normalize=True) * 100)
                print()
            print("Fitting the data finished.")
            print(np.mean(accuracy_model))

        elif self.config.get("k_fold_apply") is False:
            svm.fit(self.features, self.labels)
        else:
            print("Please select True/False to apply or not cross-validation to the training phase.")

        return svm

    def train_grid_search_svm(self):
        """
        Todo: Commenting
        :return:
        """
        print("Train an ML model with GridSearchCV")
        print("Training..")
        print()
        # define the length of parameters
        c_params = self.config.get("grid_C")
        c_params = [2 ** x for x in c_params]  # 2**x
        print("C values: {}".format(c_params))
        gamma_params = self.config.get("grid_gamma")
        gamma_params = [2 ** x for x in gamma_params]  # 2**x
        print("Gamma values: {}".format(gamma_params))
        parameters_grid = {"kernel": self.config.get("grid_kernel"),
                           "C": c_params,
                           "gamma": gamma_params,
                           "class_weight": self.config.get("grid_class_weight")
                           }
        svm = SVC(gamma="auto", probability=True)
        # get the seed from the config
        seed = self.config["random_seed"]
        # if not seed specified in the config or is None (null), get the current clock value
        # in case of clock value, convert to integer, because numpy.random.RandomState instance cannot handle float
        if seed is None or seed is "":
            seed = int(time.time())

        if self.config["gaia_imitation"] is True:
            # To be used within GridSearch if configuration's gaia imitation pratemeter is True
            inner_cv = KFold(n_splits=self.config["gaia_kfold_cv_n_splits"],
                             shuffle=self.config["gaia_kfold_shuffle"],
                             random_state=self.config["gaia_kfold_random_state"]
                             )
        else:
            # To be used within GridSearch without gaia imitation
            inner_cv = KFold(n_splits=self.config["k_fold"],
                             shuffle=self.config["shuffle"],
                             random_state=seed
                             )
        # n_jobs --> -1, means using all processors of the CPU when training the model. The default is None --> means 1
        grid = GridSearchCV(estimator=svm,
                            param_grid=parameters_grid,
                            cv=inner_cv,
                            n_jobs=self.config.get("parallel_jobs")
                            )
        grid.fit(self.features, self.labels)
        print("Best Score:")
        print(grid.best_score_)
        print()
        print("Best model:")
        print(grid.best_estimator_)
        print()
        print("Best parameters:")
        print(grid.best_params_)
        exports_dir = os.path.join(self.exports_path, self.config.get("grid_results_directory"))
        exports_path = FindCreateDirectory(exports_dir).inspect_directory()
        with open(os.path.join(exports_path, "{}.txt".format(self.class_name)), 'w+') as file:
            file.write('Grid Score - {}:'.format(self.class_name))
            file.write("\n")
            file.write("\n")
            file.write("Best Score:")
            file.write(str(grid.best_score_))
            file.write("\n")
            file.write("\n")
            file.write("Best model:")
            file.write(str(grid.best_estimator_))
            file.write("\n")
            file.write("\n")
            file.write("Best parameters:")
            file.write(str(grid.best_params_))

        return grid

    def train_neural_network(self):
        """

        :return:
        """
        print("Type of features:", type(self.features))

        early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(),
                                                                  self.exports_path,
                                                                  "nn_train", "nn_model.h5"),
                                                     monitor='val_acc',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max'
                                                     )
        # # transform to np array
        # self.features = self.features.values
        # print("Type of features:", type(self.features))
        x_train, x_test, y_train, y_test = train_test_split(self.features,
                                                            self.labels,
                                                            test_size=0.33,
                                                            random_state=42)
        instance = x_train[0]
        print('Instance X_train shape:', instance.shape)
        instance_label = y_train[0]
        print("Instance y_train shape:", instance_label.shape)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=instance.shape),
            keras.layers.Dense(instance.shape[0], activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate=.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate=.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate=.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate=.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(rate=.5),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(instance_label.shape[0])
        ])

        # Summary of the ConvNet model.
        print('Summary of the model:')
        model.summary()

        model.compile(optimizer='adam',
                      loss="categorical_crossentropy",
                      metrics=["accuracy"]
                      )
        model.fit(self.features,
                  self.labels,
                  batch_size=32,
                  epochs=100,
                  callbacks=[early_stopping, checkpoint],
                  validation_split=0.2
                  )
        scores = model.evaluate(x=x_test,
                                y=y_test,
                                verbose=1
                                )
        print("Test loss: ", scores[0])
        print("Test accuracy: ", scores[1])

    def train_randomized_search(self):
        """
        Todo: RandomizedSearch training
        :return:
        """

