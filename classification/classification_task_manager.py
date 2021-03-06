import os
from time import time
from termcolor import colored
from datetime import datetime

from ..helper_functions.utils import FindCreateDirectory, TrainingProcesses
from ..classification.classification_task import ClassificationTask
from ..helper_functions.logging_tool import LoggerSetup


validClassifiers = ["svm", "NN"]
validEvaluations = ["nfoldcrossvalidation"]


class ClassificationTaskManager:
    """
    It manages the tasks to be done based on the configuration file. It checks if the
    config keys exist in the template and are specified correctly, as well as it creates
    the relevant directories (if not exist) where the classification results will be
    stored to. Then, it extracts a list with the evaluation steps that will be followed
    with their corresponding preprocessing steps and parameters declaration for the
    classifier, and executes the classification task for each step.
    """
    def __init__(self, config, train_class, X, y, tracks, exports_path, log_level):
        """
        Args:
            config: The configuration file name.
            train_class: The class that will be trained.
            X: The already shuffled data that contain the features.
            y: The already shuffled data that contain the labels.
        """
        self.config = config
        self.train_class = train_class
        self.X = X
        self.y = y
        self.tracks = tracks
        self.exports_path = exports_path
        self.log_level = log_level

        self.exports_dir = ""
        self.results_path = ""
        self.logs_path = ""
        self.tracks_path = ""
        self.dataset_path = ""
        self.models_path = ""
        self.images_path = ""
        self.reports_path = ""

        self.logger = ""
        self.setting_logger()
        self.files_existence()
        self.config_file_analysis()

    def setting_logger(self):
        self.logger = LoggerSetup(config=self.config,
                                  exports_path=self.exports_path,
                                  name="train_model_{}".format(self.train_class),
                                  train_class=self.train_class,
                                  mode="a",
                                  level=self.log_level).setup_logger()

    def files_existence(self):
        """
        Ensure that all the folders will exist before the training process starts.
        """
        # main exports
        self.exports_dir = self.config.get("exports_directory")
        # train results exports
        self.results_path = FindCreateDirectory(self.exports_path,
                                                os.path.join(self.exports_dir, "results")).inspect_directory()
        # logs
        self.logs_path = FindCreateDirectory(self.exports_path,
                                             os.path.join(self.exports_dir, "logs")).inspect_directory()
        # tracks
        self.tracks_path = FindCreateDirectory(self.exports_path,
                                               os.path.join(self.exports_dir, "tracks_csv_format")).inspect_directory()
        # datasets
        self.dataset_path = FindCreateDirectory(self.exports_path,
                                                os.path.join(self.exports_dir, "dataset")).inspect_directory()
        # models
        self.models_path = FindCreateDirectory(self.exports_path,
                                               os.path.join(self.exports_dir, "models")).inspect_directory()
        # images
        self.images_path = FindCreateDirectory(self.exports_path,
                                               os.path.join(self.exports_dir, "images")).inspect_directory()
        # reports
        self.reports_path = FindCreateDirectory(self.exports_path,
                                               os.path.join(self.exports_dir, "reports")).inspect_directory()

    def config_file_analysis(self):
        """
        Check the keys of the configuration template file if they are set up correctly.
        """
        self.logger.info("---- CHECK FOR INAPPROPRIATE CONFIG FILE FORMAT ----")
        if "processing" not in self.config:
            self.logger.error("No preprocessing defined in config.")

        if "evaluations" not in self.config:
            self.logger.error("No evaluations defined in config.")
            self.logger.error("Setting default evaluation to 10-fold cross-validation")
            self.config["evaluations"] = {"nfoldcrossvalidation": [{"nfold": [10]}]}

        for classifier in self.config['classifiers'].keys():
            if classifier not in validClassifiers:
                self.logger.error("Not a valid classifier: {}".format(classifier))
                raise ValueError("The classifier name must be valid.")

        for evaluation in self.config['evaluations'].keys():
            if evaluation not in validEvaluations:
                self.logger.error("Not a valid evaluation: {}".format(evaluation))
                raise ValueError("The evaluation must be valid.")
        self.logger.info("No errors in config file format found.")

    def apply_processing(self):
        """
        Evaluation steps extraction and classification task execution for each step.
        """
        start_time = time()
        training_processes = TrainingProcesses(self.config).training_processes()
        self.logger.info("Classifiers detected: {}".format(self.config["classifiers"].keys()))
        for classifier in self.config["classifiers"].keys():
            print("Before Classification task: ", classifier)
            task = ClassificationTask(config=self.config,
                                      classifier=classifier,
                                      train_class=self.train_class,
                                      training_processes=training_processes,
                                      X=self.X,
                                      y=self.y,
                                      exports_path=self.exports_path,
                                      tracks=self.tracks,
                                      log_level=self.log_level
                                      )
            try:
                task.run()
            except Exception as e:
                self.logger.error('Running task failed: {}'.format(e))
                print(colored('Running task failed: {}'.format(e), "red"))
        end_time = time()

        print()
        print(colored("Last evaluation took place at: {}".format(datetime.now()), "magenta"))
        self.logger.info("Last evaluation took place at: {}".format(datetime.now()))

        # test duration
        time_duration = end_time - start_time
        classification_time = round(time_duration / 60, 2)
        return classification_time
