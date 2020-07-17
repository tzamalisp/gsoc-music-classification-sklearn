import os
import argparse
from termcolor import colored
from transformation.load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from classification.data_processing import DataProcessing
from classification.classification_task_manager import ClassificationTaskManager
from utils import load_yaml, FindCreateDirectory, TrainingProcesses


def classification_project():
    config_data = load_yaml("configuration.yaml")
    gt_files_list = ListGroundTruthFiles(config_data).list_gt_filenames()
    print(gt_files_list)
    print("LOAD GROUND TRUTH")
    for gt_file in gt_files_list:
        gt_data = GroundTruthLoad(config_data, gt_file)
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
        classification_time = class_manage.apply_processing()
        print(colored("Classification ended in {} seconds.".format(classification_time), "green"))


if __name__ == '__main__':
    classification_project()
