import os
from termcolor import colored
from transformation.load_groung_truth import ListGroundTruthFiles, GroundTruthLoad
from classification.classification_task_manager import ClassificationTaskManager
from utils import load_yaml
from transformation.load_groung_truth import DatasetExporter
import yaml


def train_class(template_data):
    config_data = load_yaml("configuration.yaml")
    gt_files_list = ListGroundTruthFiles(template_data).list_gt_filenames()
    print(gt_files_list)
    print("LOAD GROUND TRUTH")
    for gt_file in gt_files_list:
        gt_data = GroundTruthLoad(template_data, gt_file)
        # tracks shuffled and exported
        tracks_listed_shuffled = gt_data.export_gt_tracks()
        print(colored("Type of exported GT data exported: {}".format(type(tracks_listed_shuffled)), "green"))
        class_name = gt_data.export_train_class()
        template_data["class_name"] = class_name
        # project
        PROJECT_FILE_NAME = "project_{}.yaml".format(class_name)
        project_file = os.path.join(PROJECT_FILE_NAME)
        # writing project template
        with open(os.path.join(project_file), "w") as template_file:
            template_data_write = yaml.dump(template_data, template_file)

        print("First N sample of shuffled tracks: \n{}".format(tracks_listed_shuffled[:4]))

        # create the exports with the features DF, labels, and tracks together
        features, labels, tracks = DatasetExporter(config=template_data,
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

        model_manage = ClassificationTaskManager(yaml_file="configuration.yaml",
                                                 train_class=class_name,
                                                 X=features,
                                                 y=labels,
                                                 tracks=tracks)
        classification_time = model_manage.apply_processing()
        print(colored("Classification ended in {} minutes.".format(classification_time), "green"))


if __name__ == '__main__':
    train_class()
