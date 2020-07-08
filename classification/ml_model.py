import os
import pandas as pd
import numpy as np
from utils import load_yaml, FindCreateDirectory
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
# Neural Networks
import tensorflow as tf
from tensorflow import keras
import time


class TrainModel:
    """
    Todo: Commenting
    """
    def __init__(self, config, features, labels, class_name):
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

    def train_svm(self):
        """

        :return:
        """
        svc_c = 1.0
        svc_gamma = 'auto'
        # C = 2 ** C_value
        if self.config.get("svm_C") != "" and self.config.get("svm_C") is not None:
            svc_c = 2 ** self.config.get("svm_C")
        # gamma = 2 ** gamma_value
        if self.config.get("svc_gamma") != "" and self.config.get("svc_gamma") is not None:
            svc_gamma = 2 ** self.config.get("svc_gamma")

        svm = SVC(
            C=svc_c,
            kernel=self.config.get("svc_kernel"),
            gamma=svc_gamma,
            class_weight=self.config.get("svc_class_weight_balance"),
            probability=self.config.get("svc_probability")
        )

        if self.config.get("k_fold_apply") is True:
            # KFold Cross Validation approach
            X = self.features
            y = self.labels
            print("type of y:", type(y))
            kf = KFold(n_splits=self.config["gaia_kfold_cv_n_splits"],
                       shuffle=self.config["gaia_kfold_shuffle"],
                       random_state=self.config["gaia_kfold_random_state"]
                       )
            kf.split(X)

            # Initialize the accuracy of the models to blank list.
            # The accuracy of each model will be appended to this list
            accuracy_model = []

            # Iterate over each train-test split
            for train_index, test_index in kf.split(X):
                # print("TRAIN INDEX: ", train_index)
                # print("TEST INDEX: ", test_index)
                # Split train-test
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Train the model
                model = svm.fit(X_train, y_train)
                # Append to accuracy_model the accuracy of the model
                accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True) * 100)

            # Print the accuracy
            print("Accuracies in every fold iteration: {}".format(accuracy_model))
            print("Mean of all the accuracies: {}".format(sum(accuracy_model) / len(accuracy_model)))

        elif self.config.get("k_fold_apply") is False:
            svm.fit(self.features, self.labels)
        else:
            print("Please select True/False to apply or not cross-validation to the training phase.")

        return svm

    def train_neural_network(self):
        """

        :return:
        """
        print("Type of features:", type(self.features))

        early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), "exports", "nn_model.h5"),
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

    def train_grid_search(self):
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

        exports_dir = FindCreateDirectory(self.config.get("grid_results_directory")).inspect_directory()
        with open(os.path.join(exports_dir, "{}.txt".format(self.class_name)), 'w+') as file:
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

    def train_randomized_search(self):
        """
        Todo: RandomizedSearch training
        :return:
        """

