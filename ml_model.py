import os
import pandas as pd
from utils import load_yaml, FindCreateDirectory
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt




class TrainModel:
    def __init__(self, config, features, labels):
        self.config = config
        self.features = features
        self.labels = labels

    def train_svm(self):
        """
        Todo: SVM training
        """

    def train_neural_network(self):
        """
        Todo: Neural Network training
        """

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
        grid = GridSearchCV(estimator=svm, param_grid=parameters_grid, cv=5)
        grid.fit(self.features, self.labels)
        print("Best Score:")
        print(grid.best_score_)
        print()
        print("Best model:")
        print(grid.best_estimator_)
        print()
        print("Best parameters:")
        print(grid.best_params_)

        return grid

    def train_randomized_search(self):
        """
        Todo: GridSearch training
        """


class Evaluation:
    def __init__(self, model, x_data, y_data, model_name):
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.model_name = model_name

    def model_evaluation(self):
        """A function to compute the evaluation based on
        Parameters:
        model: the model to run predictions
        X_data (numpy array): the test data to provide predictions from
        y_data (pd.Series or numpy array): the labels data
        model_name (str): The name of the model that will be evaluated

        """
        print("Model to evaluate:", self.model_name)
        print("Features shape:", self.x_data.shape)
        print("Labels shape:", self.y_data.shape)
        print("Model classes:", self.model.classes_)
        print()

        # predictions
        print("PREDICTIONS - {}".format(self.model_name))
        predictions = self.model.predict(self.x_data)
        print("{} - Confusion matrix:".format(self.model_name))
        print(confusion_matrix(self.y_data, predictions))
        print("Confusion Matrix: Each class in Percent of recall.")
        cm = confusion_matrix(self.y_data, predictions)
        cm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100
        print(cm)
        print("Type of CM:", type(cm))
        print()
        print("Classification Report:")
        print(classification_report(self.y_data, predictions))
        print()

        print("PROBABILITIES - {}".format(self.model_name))
        # probabilities for each prediction
        prob_predictions = self.model.predict_proba(self.x_data)
        prob_predictions = prob_predictions * 100  # multiply * 100 for better resutls
        # df_prob_gsvc_pca_test.drop(df.index, inplace=True)  # if DF exists, empty it
        df_prob_predictions = pd.DataFrame(prob_predictions)
        print("Head of the predictions probabilities DF:")
        print(df_prob_predictions.head())
        print()
        # transform the predictions and the test data for the
        # DF concatenation with the probabilities DF
        series_predictions = pd.Series(predictions)
        y_data_reindexed = self.y_data.reset_index(drop=True)
        print("Type of the re-indexed y_data:", type(y_data_reindexed))
        print()
        # concatenate
        df_prob_predictions = pd.concat([df_prob_predictions, series_predictions, y_data_reindexed],
                                        axis=1, ignore_index=False)

        df_prob_predictions.columns = ["prob_pred_dance", "prob_pred_not_dance", "prediction", "true"]
        print("Final DF:")
        print(df_prob_predictions.head())
        print()

        print("MEAN ACCURACIES IN THE PROBABILITIES PREDICTIONS - {}".format(self.model_name))
        tp_values = []
        tn_values = []
        fp_values = []
        fn_values = []

        for index, row in df_prob_predictions.iterrows():
            if row["true"] == row["prediction"]:
                if row["prediction"] == "danceable":
                    tp_values.append(row["prob_pred_dance"])
                elif row["prediction"] == "not_danceable":
                    tn_values.append(row["prob_pred_not_dance"])
            if row["true"] != row["prediction"]:
                if row["prediction"] == "danceable":
                    fn_values.append(row["prob_pred_dance"])
                elif row["prediction"] == "not_danceable":
                    fp_values.append(row["prob_pred_not_dance"])

        # TP
        if len(tp_values) is 0:
            tp_mean = 0.0
        else:
            tp_mean = sum(tp_values) / len(tp_values)
        print("TP mean:", tp_mean)
        # FP
        if len(fp_values) is 0:
            fp_mean = 0.0
        else:
            fp_mean = sum(fp_values) / len(fp_values)
        print("FP mean:", fp_mean)
        # TN
        if len(tn_values) is 0:
            tn_mean = 0.0
        else:
            tn_mean = sum(tn_values) / len(tn_values)
        print("TN mean:", tn_mean)
        # FN
        if len(fn_values) is 0:
            fn_mean = 0.0
        else:
            fn_mean = sum(fn_values) / len(fn_values)
        print("FN mean:", fn_mean)
