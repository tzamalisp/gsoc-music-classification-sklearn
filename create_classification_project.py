import os
import argparse
from pprint import pprint
from utils import load_yaml
import yaml
import time
from transformation.load_ground_truth import ListGroundTruthFiles
from classification.train_class import train_class


def create_classification_project(ground_truth_directory, project_file, exports_directory, logging, seed, jobs, verbose, exports_path):
    """
    Args:
        ground_truth_directory:
        project_file:
        exports_directory:
        logging:
        seed:
        jobs:
        verbose:
        exports_path:

    """
    try:
        project_template = load_yaml("configuration_template.yaml")
    except Exception as e:
        print('Unable to open project configuration template:', e)
        raise

    # print("BEFORE:")
    # print("Type of congig template:", type(project_template))
    print("-------------------------------------------------------")
    print()
    if seed is None:
        seed = time.time()

    print("Seed argument: {}".format(seed))

    project_template["ground_truth_directory"] = ground_truth_directory
    project_template["project_file"] = project_file
    project_template["logging_level"] = logging
    project_template["seed"] = seed
    project_template["parallel_jobs"] = jobs
    project_template["verbose"] = verbose

    # if empty, path is declared as the app's main directory
    if exports_path is None:
        exports_path = os.getcwd()

    print("Exports path: {}".format(exports_path))
    project_template["exports_path"] = exports_path

    print()
    print()
    print("-------------------------------------------------------")
    # print("AFTER:")
    # pprint(project_template)

    gt_files_list = ListGroundTruthFiles(project_template).list_gt_filenames()
    print(gt_files_list)
    print("LOAD GROUND TRUTH")
    for gt_file in gt_files_list:
        train_class(project_template, gt_file, exports_directory, logging)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates a project configuration file given a filelist, a groundtruth file, '
                    'and the directories to store the datasets and the results files. '
                    'The script has a parameter to specify the project template to use. '
                    'If it is not specified, it will try to guess the appropriated one from the '
                    'essentia version found on the descriptor files.')

    parser.add_argument('-g', '--groundtruth',
                        dest="ground_truth_directory",
                        default="datasets",
                        help='Path of the dataset directory containing the groundtruth file/s.',
                        required=True)

    parser.add_argument('-f', '--file',
                        dest="project_file",
                        default="project",
                        help='Name prefix of the project configuration file (.yaml) will be stored.')

    parser.add_argument('-e', '--exportsdir',
                        dest="exports_directory",
                        help='Path the exports of the project will be stored.')

    parser.add_argument('-p', '--path',
                        dest='exports_path',
                        help='Path where the project results will be stored. If empty, the results will be saved in '
                             'app directory')

    parser.add_argument('-l', '--logging',
                        default=1,
                        help='Path where the result files will be stored.',
                        type=int)

    parser.add_argument('-s', '--seed',
                        default=None,
                        help='Seed used to generate the shuffled dataset applied later to folding.',
                        type=int)

    parser.add_argument('-j', '--jobs',
                        default=-1,
                        help='Parallel jobs. Set to -1 to use all the available cores',
                        type=int)
    parser.add_argument('-v', '--verbose',
                        default=1,
                        help="Controls the verbosity: the higher, the more messages.",
                        type=int)

    args = parser.parse_args()

    create_classification_project(args.ground_truth_directory, args.project_file,
                                  args.exports_directory, logging=args.logging, seed=args.seed, jobs=args.jobs,
                                  verbose=args.verbose, exports_path=args.exports_path)
