""" This file contains the Evaluation class script.

The Evaluation class uses the model that is trained and evaluates it based on the test data that is inserted
into it. A report file is exported locally, which contains the classification report and the confusion
matrix of the model that is tested on the data. It also includes the mean of the probabilities that the
decision function took and correctly decided which class each instance corresponds to.

    Typical usage example:

    evaluation_object = Evaluation(config_data, model_object, X_test, y_test, class_name_string)
    evaluate_model = evaluation_object.model_evaluation()

"""
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from utils import FindCreateDirectory
from datetime import datetime
from ml_logger import LoggerSetup
from datetime import datetime


class Evaluation:
    """It evaluates the model by testing on some instances and using features and labels data.

    The class uses as inputs, the data from the configuration file, the testing instances data (features and labels),
    the trained model object, and a string that describes the class that will be evaluated.

    The model_evaluation() method does the evaluation of the trained model for the specified class testing data and
    extracts a txt file that contains the classification report, the confusion matrix, and the mean decision
    probabilities of all the corrected classified instances.

    Attributes:
        config: A dictionary that includes the data from the configuration file.
        model: An object of the trained model that will be evaluated.
        x_data: The features data of the testing instances.
        y_data: The labels of the testing instances.
        class_name: A string that describes the class that will be evaluated.
    """
    def __init__(self, config, model, x_data, y_data, class_name):
        """
        Inits the Evaluation with the corresponding parameters.

        Args:
            config (dict): Configuration data that is load from the conf file in form of a dictionary.
            model (object): The scikit-learn model object to run predictions.
            x_data (object): The test data to provide predictions from in NumPy array.
            y_data (object): The labels data in pd.Series or numpy array.
            class_name (str): The name of the model that will be evaluated.
        """
        self.config = config
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.class_name = class_name

    def model_evaluation(self):
        """
        Computes the evaluation of the testing instances and extracts the evaluation report .txt file.

        Returns:

        """
        # logging
        # logs_dir = FindCreateDirectory(self.config.get("log_directory")).inspect_directory()
        # path_logger = os.path.join(logs_dir, "evaluation.log")
        log_eval = LoggerSetup(name="evaluation_logger",
                               log_file="evaluation.log",
                               level=self.config.get("logging_level")
                               )
        logger_eval = log_eval.setup_logger()

        logger_eval.info("Model to evaluate: {}".format(self.config.get("train_kind")))
        logger_eval.info("Class to evaluate: {}".format(self.class_name))
        # logger_eval.info("Features shape: {}".format(self.x_data.shape))
        # logger_eval.info("Labels shape: {}".format(self.y_data.shape))
        # logger_eval.info("Model classes: {}".format(self.model.classes_))

        # predictions
        logger_eval.info("PREDICTIONS - {}".format(self.config.get("train_kind")))
        predictions = self.model.predict(self.x_data)
        # confusion matrix
        logger_eval.debug("{} - Confusion matrix:".format(self.config.get("train_kind")))
        logger_eval.debug("\n{}".format(confusion_matrix(self.y_data, predictions)))
        # confusion matrix in proportion to the whole class items
        logger_eval.info("Confusion Matrix: Each class in Percent of recall.")
        cm = confusion_matrix(self.y_data, predictions)
        cm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100
        logger_eval.info("\n{}".format(cm))
        logger_eval.info("Type of CM: {}".format(type(cm)))
        # classification report
        logger_eval.info("Classification Report:")
        logger_eval.info("\n{}".format(classification_report(self.y_data, predictions)))
        # probabilities
        logger_eval.info("PROBABILITIES - {}".format(self.config.get("train_kind")))
        # probabilities for each prediction
        prob_predictions = self.model.predict_proba(self.x_data)
        prob_predictions = prob_predictions * 100  # multiply * 100 for better resutls
        logger_eval.info("PREDICTIONS: \n{}".format(predictions[:10]))
        logger_eval.info("TRUE: \n{}".format(self.y_data[:10]))
        logger_eval.info("PREDICTIONS PROBABILITIES: \n{}".format(prob_predictions[:10]))
        # df_prob_gsvc_pca_test.drop(df.index, inplace=True)  # if DF exists, empty it
        df_prob_predictions = pd.DataFrame(prob_predictions)
        logger_eval.info("Head of the predictions probabilities DF:")
        logger_eval.info("\n{}".format(df_prob_predictions.head()))
        # transform the predictions and the test data for the
        # DF concatenation with the probabilities DF
        df_predictions = pd.DataFrame(predictions, columns=["prediction"])
        y_data_re_indexed = self.y_data.reset_index(drop=True)
        logger_eval.info("Type of the re-indexed y_data: {}".format(type(y_data_re_indexed)))

        # concatenate
        df_prob_predictions = pd.concat([df_prob_predictions, df_predictions, y_data_re_indexed],
                                        axis=1, ignore_index=False)
        logger_eval.info("DF BEFORE COLUMN TRANSFORMATION")
        logger_eval.info("\n{}".format(df_prob_predictions.head()))
        # df_class_columns = []
        # for item in self.model.classes_:
        #     df_class_columns.append(item)
        # df_class_columns.append("prediction")
        # df_class_columns.append("true")
        df_prob_predictions.rename(columns={self.class_name: "true"}, inplace=True)
        # df_prob_predictions.columns = ["prob_pred_dance", "prob_pred_not_dance", "prediction", "true"]
        # df_prob_predictions.columns = df_class_columns
        logger_eval.info("FINAL DF AFTER COLUMN TRANSFORMATION:")
        logger_eval.info("\n{}".format(df_prob_predictions.head()))

        logger_eval.info("MEAN ACCURACIES IN THE PROBABILITIES PREDICTIONS - {}".format(self.config.get("train_kind")))
        report_list = []
        for item in self.model.classes_:
            pred_mean_truth_pred = []
            for index, row in df_prob_predictions.iterrows():
                if row["true"] == row["prediction"]:
                    if row["prediction"] == item:
                        pred_mean_truth_pred.append(np.max(row[:-2].values))
                        # pred_mean_truth_pred.append(row[item])
            length_report = "{} - length of list: {}".format(item, len(pred_mean_truth_pred))
            logger_eval.info(length_report)
            report_list.append(length_report)
            if len(pred_mean_truth_pred) is 0:
                mean_value = 0.0
                mean_report = "{} - Mean: {}".format(item, mean_value)
                logger_eval.info(mean_report)
                report_list.append(mean_report)
            else:
                mean_value = sum(pred_mean_truth_pred) / len(pred_mean_truth_pred)
                mean_report = "{} - Mean: {}".format(item, mean_value)
                logger_eval.info(mean_report)
                report_list.append(mean_report)
            logger_eval.info("Next class:")
        exports_dir = FindCreateDirectory(self.config.get("evaluations_directory")).inspect_directory()
        # take current date and convert to string
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d")
        datetime_str_verbose = now.strftime("%Y-%m-%d, %H:%M:%S")
        print("Creating report file..")
        with open(os.path.join(exports_dir,
                               "{}_{}_{}.txt".format(self.class_name,
                                                     self.config.get("train_kind"),
                                                     datetime_str)), 'w+') as file:
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
            file.write("Date of execution: {}".format(datetime_str_verbose))
            file.close()
        print('Evaluation file for class {} is created successfully.'.format(self.config.get("class_name_train")))
