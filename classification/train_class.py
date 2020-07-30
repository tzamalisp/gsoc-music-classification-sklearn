import os
from termcolor import colored
from transformation.load_groung_truth import GroundTruthLoad
from classification.classification_task_manager import ClassificationTaskManager
from transformation.load_groung_truth import DatasetExporter
import yaml


def train_class(config, gt_file):
    exports_path = config["exports_path"]

    gt_data = GroundTruthLoad(config, gt_file)
    # tracks shuffled and exported
    tracks_listed_shuffled = gt_data.export_gt_tracks()
    print(colored("Type of exported GT data exported: {}".format(type(tracks_listed_shuffled)), "green"))

    # class to train
    class_name = gt_data.export_train_class()
    config["class_name"] = class_name

    # save project file
    project_file_name_save = "{}_{}.yaml".format(config["project_file"], class_name)
    project_file_save_path = os.path.join(exports_path, project_file_name_save)
    with open(os.path.join(project_file_save_path), "w") as template_file:
        template_data_write = yaml.dump(config, template_file)

    print("First N sample of shuffled tracks: \n{}".format(tracks_listed_shuffled[:4]))

    # create the exports with the features DF, labels, and tracks together
    features, labels, tracks = DatasetExporter(config=config,
                                               tracks_list=tracks_listed_shuffled,
                                               train_class=class_name,
                                               exports_path=exports_path
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

    model_manage = ClassificationTaskManager(config=config,
                                             train_class=class_name,
                                             X=features,
                                             y=labels,
                                             tracks=tracks,
                                             exports_path=exports_path)
    classification_time = model_manage.apply_processing()
    print(colored("Classification ended in {} minutes.".format(classification_time), "green"))

