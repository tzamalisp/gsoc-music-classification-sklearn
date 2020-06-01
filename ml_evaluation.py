import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from utils import FindCreateDirectory
from datetime import datetime


class Evaluation:
    def __init__(self, config, model, x_data, y_data):
        self.config = config
        self.model = model
        self.x_data = x_data
        self.y_data = y_data

    def model_evaluation(self):
        """
        A function to compute the evaluation based on
        Parameters:
        model: the model to run predictions
        X_data (numpy array): the test data to provide predictions from
        y_data (pd.Series or numpy array): the labels data
        config.get("train_kind") (str): The name of the model that will be evaluated
        """
        class_to_evaluate = self.config.get("class_name_train")
        if class_to_evaluate.startswith("timbre"):
            class_to_evaluate = "timbre"
        print("Model to evaluate:", self.config.get("train_kind"))
        print("Class to evaluate:", class_to_evaluate)
        print("Features shape:", self.x_data.shape)
        print("Labels shape:", self.y_data.shape)
        print("Model classes:", self.model.classes_)
        print()

        # predictions
        print("PREDICTIONS - {}".format(self.config.get("train_kind")))
        predictions = self.model.predict(self.x_data)
        print("{} - Confusion matrix:".format(self.config.get("train_kind")))
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

        print("PROBABILITIES - {}".format(self.config.get("train_kind")))
        # probabilities for each prediction
        prob_predictions = self.model.predict_proba(self.x_data)
        prob_predictions = prob_predictions * 100  # multiply * 100 for better resutls
        print("PREDICTIONS\n", predictions[:10])
        print("TRUE\n", self.y_data[:10])
        print("PREDICTIONS PROBA\n", prob_predictions[:10])
        # df_prob_gsvc_pca_test.drop(df.index, inplace=True)  # if DF exists, empty it
        df_prob_predictions = pd.DataFrame(prob_predictions)
        print("Head of the predictions probabilities DF:")
        print(df_prob_predictions.head())
        print()
        # transform the predictions and the test data for the
        # DF concatenation with the probabilities DF
        df_predictions = pd.DataFrame(predictions, columns=["prediction"])
        y_data_re_indexed = self.y_data.reset_index(drop=True)
        print("Type of the re-indexed y_data:", type(y_data_re_indexed))
        print()
        # concatenate
        df_prob_predictions = pd.concat([df_prob_predictions, df_predictions, y_data_re_indexed],
                                        axis=1, ignore_index=False)
        print("AAAAAAAAAAAAAA")
        print(df_prob_predictions.head())
        # df_class_columns = []
        # for item in self.model.classes_:
        #     df_class_columns.append(item)
        # df_class_columns.append("prediction")
        # df_class_columns.append("true")
        df_prob_predictions.rename(columns={class_to_evaluate: "true"}, inplace=True)
        # df_prob_predictions.columns = ["prob_pred_dance", "prob_pred_not_dance", "prediction", "true"]
        # df_prob_predictions.columns = df_class_columns
        print("Final DF of predictions:")
        print(df_prob_predictions.head())
        print()

        print("MEAN ACCURACIES IN THE PROBABILITIES PREDICTIONS - {}".format(self.config.get("train_kind")))
        report_list = []
        for item in self.model.classes_:
            pred_mean_truth_pred = []
            for index, row in df_prob_predictions.iterrows():
                if row["true"] == row["prediction"]:
                    if row["prediction"] == item:
                        pred_mean_truth_pred.append(np.max(row[:-2].values))
                        # pred_mean_truth_pred.append(row[item])
            length_report = "{} - length of list: {}".format(item, len(pred_mean_truth_pred))
            print(length_report)
            report_list.append(length_report)
            if len(pred_mean_truth_pred) is 0:
                mean_value = 0.0
                mean_report = "{} - Mean: {}".format(item, mean_value)
                print(mean_report)
                report_list.append(mean_report)
            else:
                mean_value = sum(pred_mean_truth_pred) / len(pred_mean_truth_pred)
                mean_report = "{} - Mean: {}".format(item, mean_value)
                print(mean_report)
                report_list.append(mean_report)
            print()
        exports_dir = FindCreateDirectory(self.config.get("evaluations_directory")).inspect_directory()

        with open(os.path.join(exports_dir,
                               "{}_{}_classification_report.txt".format(self.config.get("class_name_train"),
                                                                        self.config.get("train_kind"))), 'w+') as file:
            file.write('Classification Report:')
            file.write(str(classification_report(self.y_data, predictions)))
            file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write('Confusion Matrix:')
            file.write('\n')
            file.write(str(confusion_matrix(self.y_data, predictions)))
            file.write('\n')
            file.write('\n')
            file.write('Confusion Matrix (Proportion of TP, FP, TN, FN values):')
            file.write('\n')
            file.write(str(cm))
            file.write('\n')
            file.write('\n')
            file.write("DF part with predictions probabilities:")
            file.write('\n')
            file.write(str(df_prob_predictions.head()))
            file.write('\n')
            file.write('\n')
            file.write('Mean of corrected predictions:')
            file.write('\n')
            for item in report_list:
                file.write(str(item))
                file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write("Date of execution:".format(datetime.now()))
            file.close()
        print('Evaluation file for class {} is created successfully.'.format(self.config.get("class_name_train")))

        tp_values = []
        tn_values = []
        fp_values = []
        fn_values = []

        # for index, row in df_prob_predictions.iterrows():
        #     if row["true"] == row["prediction"]:
        #         if row["prediction"] == "danceable":
        #             tp_values.append(row["prob_pred_dance"])
        #         elif row["prediction"] == "not_danceable":
        #             tn_values.append(row["prob_pred_not_dance"])
        #     if row["true"] != row["prediction"]:
        #         if row["prediction"] == "danceable":
        #             fn_values.append(row["prob_pred_dance"])
        #         elif row["prediction"] == "not_danceable":
        #             fp_values.append(row["prob_pred_not_dance"])
        #
        # # TP
        # if len(tp_values) is 0:
        #     tp_mean = 0.0
        # else:
        #     tp_mean = sum(tp_values) / len(tp_values)
        # print("TP mean:", tp_mean)
        # # FP
        # if len(fp_values) is 0:
        #     fp_mean = 0.0
        # else:
        #     fp_mean = sum(fp_values) / len(fp_values)
        # print("FP mean:", fp_mean)
        # # TN
        # if len(tn_values) is 0:
        #     tn_mean = 0.0
        # else:
        #     tn_mean = sum(tn_values) / len(tn_values)
        # print("TN mean:", tn_mean)
        # # FN
        # if len(fn_values) is 0:
        #     fn_mean = 0.0
        # else:
        #     fn_mean = sum(fn_values) / len(fn_values)
        # print("FN mean:", fn_mean)
