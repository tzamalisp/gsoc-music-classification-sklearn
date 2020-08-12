import os
import yaml


def load_yaml(path_file):
    """
    Todo: add comments, docstring info, etc.
    Returns:
        The configuration data loaded from the template.
    """
    try:
        with open(os.path.join(os.path.abspath(os.getcwd()), path_file)) as file:
            config_data = yaml.load(file, Loader=yaml.FullLoader)
            if isinstance(config_data, dict):
                return config_data
            else:
                return None
    except ImportError:
        print("WARNING: could not import yaml module")
        return None


class DfChecker:
    """

    """
    def __init__(self, df_check):
        """
        Args:
            df_check:
        """
        self.df_check = df_check

    def check_df_info(self):
        """
        Prints information about the Pandas DataFrame that is generated from the relevant process.
        """
        print("Features DataFrame head:")
        print(self.df_check.head())
        print()
        print("Information:")
        print(self.df_check.info())
        print()
        print("Shape:", self.df_check.shape)
        print("Number of columns:", len(list(self.df_check.columns)))

        if "category" in self.df_check.columns:
            print("Track categories distribution:")
            print(self.df_check["category"].value_counts())


class FindCreateDirectory:
    def __init__(self, exports_path, directory):
        self.exports_path = exports_path
        self.directory = directory

    def inspect_directory(self):
        # find dynamically the current script directory
        # path_app = os.path.join(os.path.abspath(os.getcwd()))
        full_path = os.path.join(self.exports_path, self.directory)
        # create path directories if not exist --> else return the path
        os.makedirs(full_path, exist_ok=True)
        return full_path


class LogsDeleter:
    def __init__(self, config, train_class):
        self.config = config
        self.train_class = train_class

    def delete_logs(self):
        # delete logs for specific model and class on a new run
        if self.config["delete_logs"] is True:
            print("Evaluation logs deletion is turned to ON.")
            dir_name = os.path.join(os.getcwd(), "evaluations")
            evaluations_list = os.listdir(dir_name)
            for item in evaluations_list:
                if item.endswith(".txt"):
                    if item.startswith("{}_{}".format(self.train_class, self.config["train_kind"])):
                        os.remove(os.path.join(dir_name, item))
            print("Previous evaluation logs deleted successfully.")
        else:
            print("Evaluation logs deletion is turned to OFF.")


def change_weights_values(i):
    if i is True:
        return "balanced"
    elif i is False:
        return None
    return i


class TrainingProcesses:
    def __init__(self, config):
        self.config = config

    def training_processes(self):
        """
        Returns:
            A list of the processes that have been identified with the corresponding parameter grid.
        """
        evaluations = self.config["evaluations"]["nfoldcrossvalidation"]
        print("Evaluations countered: {}".format(len(evaluations)))
        evaluation_counter = 0
        trainings_counted = 0
        processes = []
        for evaluation in evaluations:
            for nfold_number in evaluation["nfold"]:
                classifiers = self.config["classifiers"]["svm"]
                for classifier in classifiers:
                    for pre_processing in classifier["preprocessing"]:
                        for clf_type in classifier["type"]:
                            if clf_type == "C-SVC":
                                process_dict = dict()
                                process_dict["Evaluation"] = evaluation_counter
                                # classifier
                                process_dict["classifier"] = clf_type
                                # pre-processing
                                process_dict["preprocess"] = pre_processing
                                # kernel
                                kernel = classifier["kernel"]
                                process_dict["kernel"] = [i.lower() for i in kernel]
                                # C
                                c = classifier["C"]
                                process_dict["C"] = [2 ** x for x in c]  # 2 ** c
                                # gamma
                                gamma = classifier["gamma"]
                                process_dict["gamma"] = [2 ** x for x in gamma]  # 2 ** gamma
                                # class weights
                                balance_classes = classifier["balanceClasses"]
                                process_dict["balanceClasses"] = [change_weights_values(i) for i in balance_classes]
                                processes.append(process_dict)
                                # n_fold
                                process_dict["n_fold"] = nfold_number
                                # increase counter by 1
                                trainings_counted += 1
            # increase evaluation counter by 1
            evaluation_counter += 1

        print("Trainings to be applied: {}".format(trainings_counted))

        return processes
