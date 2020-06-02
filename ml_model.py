import os
import pandas as pd
import numpy as np
from utils import load_yaml, FindCreateDirectory
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# Neural Networks
import tensorflow as tf
from tensorflow import keras


class TrainModel:
    def __init__(self, config, features, labels, class_name):
        self.config = config
        self.features = features
        self.labels = labels
        self.class_name = class_name

    def train_svm(self):
        """
        Todo: SVM training
        """
        svm = SVC(gamma="auto", probability=True)
        svm.fit(self.features, self.labels)

        return svm

    def train_neural_network(self):
        """
        Todo: Neural Network training
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
        Todo: GridSearch training
        """
        print("Train an ML model with GridSearchCV")
        print("Training..")
        print()
        # define the length of parameters
        parameters_grid = {"kernel": self.config.get("grid_kernel"),
                           "C": self.config.get("grid_C"),
                           "gamma": self.config.get("grid_gamma"),
                           "class_weight": self.config.get("grid_class_weight")
                           }
        svm = SVC(gamma="auto", probability=True)
        # n_jobs --> -1, means using all processors of the CPU when training the model. The defaults None: means 1
        grid = GridSearchCV(estimator=svm, param_grid=parameters_grid, cv=5, n_jobs=self.config.get("parallel_jobs"))
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
        """

