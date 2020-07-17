import os
import logging
from time import time
from pprint import pprint
from termcolor import colored
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from classification.data_processing import DataProcessing
from classification.classification_task import ClassificationTask
from transformation.load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from datetime import datetime

log = logging.getLogger('classification.ClassificationTaskManager')

validClassifiers = ['NN', 'svm']
validEvaluations = ['nfoldcrossvalidation']


class ClassificationTaskManager:
    """

    """
    def __init__(self, yaml_file, train_class, X, y, tracks):
        """

        :param yaml_file: The configuration file name
        :param train_class: The class that will be trained
        :param X: The already shuffled data that contain the features
        :param y: The already shuffled data that contain the labels
        """
        self.yaml_file = yaml_file
        self.train_class = train_class
        self.X = X
        self.y = y
        self.tracks = tracks

        self.config = ""
        self.exports_path = ""

        self.results_path = ""
        self.logs_path = ""
        self.tracks_path = ""
        self.dataset_path = ""

        self.load_config()
        self.files_existence()
        self.config_file_analysis()

    def load_config(self):
        try:
            self.config = load_yaml(self.yaml_file)
        except Exception as e:
            print('Unable to open project configuration file:', e)
            raise

    def files_existence(self):
        """
        Ensure that all the folders will exist before the training process starts
        :return:
        """
        # main exports
        exports_dir = "{}_{}".format(self.config.get("exports_directory"), self.train_class)
        self.exports_path = FindCreateDirectory(exports_dir).inspect_directory()
        # train results exports
        self.results_path = FindCreateDirectory(os.path.join(self.exports_path, "results")).inspect_directory()
        # logs
        self.logs_path = FindCreateDirectory(os.path.join(self.exports_path, "logs")).inspect_directory()
        # tracks
        self.tracks_path = FindCreateDirectory(os.path.join(self.exports_path, "tracks_csv")).inspect_directory()
        # datasets
        self.dataset_path = FindCreateDirectory(os.path.join(self.exports_path, "dataset")).inspect_directory()

    def config_file_analysis(self):

        if 'processing' not in self.config:
            log.warning('No preprocessing defined in {}.'.format(self.yaml_file))

        if 'evaluations' not in self.config:
            log.warning('No evaluations defined in {}.'.format(self.yaml_file))
            log.warning('Setting default evaluation to 10-fold cross-validation')
            self.config['evaluations'] = {'nfoldcrossvalidation': [{'nfold': [10]}]}

        for classifier in self.config['classifiers'].keys():
            if classifier not in validClassifiers:
                log.warning('Not a valid classifier: {}'.format(classifier))
                raise ValueError('The classifier name must be valid.')

        for evaluation in self.config['evaluations'].keys():
            if evaluation not in validEvaluations:
                log.warning('Not a valid evaluation: {}'.format(evaluation))
                raise ValueError("The evaluation must be valid.")

    def apply_processing(self):
        start_time = time()

        training_processes = TrainingProcesses(self.config).training_processes()
        print(colored("Classifiers detected: {}".format(self.config["classifiers"].keys()), "green"))
        for classifier in self.config['classifiers'].keys():
            print("Before Classification task: ", classifier)
            task = ClassificationTask(config=self.config,
                                      classifier=classifier,
                                      train_class=self.train_class,
                                      training_processes=training_processes,
                                      X=self.X,
                                      y=self.y,
                                      exports_path=self.exports_path,
                                      tracks=self.tracks
                                      )
            # try:
            #     task.run()
            # except Exception as e:
            #     log.error('Running task failed: %s' % e)
            task.run()

        end_time = time()

        print()
        print(colored("Last evaluation took place at: {}".format(datetime.now()), "magenta"))

        # test duration
        return end_time - start_time


if __name__ == '__main__':
    config_data = load_yaml("configuration.yaml")
    gt_data = GroundTruthLoad(config_data, "groundtruth.yaml")
    df_fg_data = gt_data.export_gt_tracks()
    print(colored("Type of exported GT data exported: {}".format(type(df_fg_data)), "green"))
    class_name = gt_data.export_train_class()

    data_processing_obj = DataProcessing(config=config_data,
                                         dataset=df_fg_data,
                                         class_name=class_name
                                         )
    tracks_shuffled = data_processing_obj.shuffle_tracks_data()
    print(colored("SHUFFLED TRACKS:", "green"))
    print(tracks_shuffled[:4])
    print()
    X, y = data_processing_obj.exporting_classification_data()

    class_manage = ClassificationTaskManager(yaml_file="configuration.yaml",
                                             train_class=class_name,
                                             X=X,
                                             y=y,
                                             tracks=tracks_shuffled)
    class_manage.apply_processing()

