import os
import argparse
from termcolor import colored
from transformation.load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from classification.data_processing import DataProcessing
from classification.classification_task_manager import ClassificationTaskManager
from utils import load_yaml, FindCreateDirectory, TrainingProcesses
from transformation.load_groung_truth import DatasetDFCreator

def classification_project():
    config_data = load_yaml("configuration.yaml")
    gt_files_list = ListGroundTruthFiles(config_data).list_gt_filenames()
    print(gt_files_list)
    print("LOAD GROUND TRUTH")
    for gt_file in gt_files_list:
        gt_data = GroundTruthLoad(config_data, gt_file)
        tracks_listed_shuffled = gt_data.export_gt_tracks()
        print(colored("Type of exported GT data exported: {}".format(type(tracks_listed_shuffled)), "green"))
        class_name = gt_data.export_train_class()
        print("First N sample of shuffled tracks: \n{}".format(tracks_listed_shuffled[:4]))

        # data_processing_obj = DataProcessing(config=config_data,
        #                                      dataset=tracks_listed_shuffled,
        #                                      class_name=class_name
        #                                      )

        # create DF with the features, labels, and tracks together
        features, labels, tracks = DatasetDFCreator(config=config_data,
                                                    tracks_list=tracks_listed_shuffled,
                                                    train_class=class_name
                                                    ).create_df_tracks()
        print(colored("Types of exported files from GT:", "cyan"))
        print("Type of features: {}".format(type(features)))
        print("Type of labels: {}".format(type(labels)))
        print("Type of Tracks: {}".format(type(tracks)))
        print()
        print(colored("Small previews:", "cyan"))
        print(colored("FEATURES", "magenta"))
        print(features.head(10))
        print(colored("LABELS", "magenta"))
        print(labels[:10])
        print(colored("TRACKS:", "magenta"))
        print(tracks[:10])

        # X, y = data_processing_obj.exporting_classification_data()
        #
        model_manage = ClassificationTaskManager(yaml_file="configuration.yaml",
                                                 train_class=class_name,
                                                 X=features,
                                                 y=labels,
                                                 tracks=tracks)
        classification_time = model_manage.apply_processing()
        print(colored("Classification ended in {} seconds.".format(classification_time), "green"))


if __name__ == '__main__':
    classification_project()
