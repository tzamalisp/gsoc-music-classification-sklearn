import os
import json
from pprint import pprint
from termcolor import colored

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from transformation.transform import Transform
from utils import load_yaml, FindCreateDirectory, TrainingProcesses


class TrainGridClassifier:
    def __init__(self, config, class_name, X, y, tr_processes, n_fold, exports_path):
        self.config = config
        self.class_name = class_name
        self.X = X
        self.y = y
        self.tr_processes = tr_processes
        self.n_fold = n_fold
        self.exports_path = exports_path

        self.best_models_list = []
        self.train_grid_search_svm()

    def train_grid_search_svm(self):
        process_counter = 1
        for tr_process in self.tr_processes:
            print("Train process {} - {}".format(process_counter, tr_process))
            # evaluate()
            if tr_process["classifier"]:
                print("CLASSIFIER", tr_process["classifier"])
                # transformation of the data
                X_transformed = Transform(config=self.config,
                                          df=self.X,
                                          process=tr_process["preprocess"],
                                          exports_path=self.exports_path
                                          ).post_processing()

                # define the length of parameters
                parameters_grid = {'kernel': tr_process["kernel"],
                                   'C': tr_process["C"],
                                   'gamma': tr_process["gamma"],
                                   'class_weight': tr_process["balanceClasses"]
                                   }
                # initiate SVM classifier object
                if tr_process["classifier"] == "C-SVC":
                    svm = SVC(gamma="auto", probability=True)
                else:
                    raise ValueError('The classifier name must be valid.')

                # inner with K-Fold cross-validation declaration
                random_seed = None
                shuffle = self.config["k_fold_shuffle"]
                if shuffle is True:
                    random_seed = self.config["k_fold_random_seed"]
                elif shuffle is False:
                    random_seed = None
                print("Fitting the data to the classifier with K-Fold cross-validation..")
                inner_cv = KFold(n_splits=self.n_fold,
                                 shuffle=shuffle,
                                 random_state=random_seed
                                 )
                # initiate GridSearch Object
                gsvc = GridSearchCV(estimator=svm,
                                    param_grid=parameters_grid,
                                    cv=inner_cv,
                                    n_jobs=self.config["parallel_jobs"],
                                    verbose=self.config["grid_verbose"]
                                    )

                print(colored("Shape of X before train: {}".format(X_transformed.shape), "green"))
                print("Fitting the data to the model:")
                gsvc.fit(X_transformed, self.y)

                # print(gsvc.cv_results_["params"])
                print(gsvc.best_score_)
                print(gsvc.best_estimator_)
                print(gsvc.best_params_)
                print("Counted evaluations in this GridSearch process: {}".format(len(gsvc.cv_results_["params"])))

                exports_dir = "{}_{}".format(self.config.get("exports_directory"), self.class_name)
                exports_path = FindCreateDirectory(exports_dir).inspect_directory()

                # save best results for each train process
                grid_results_dir = os.path.join(exports_path, "results")
                grid_results_path = FindCreateDirectory(grid_results_dir).inspect_directory()
                results_best_dict_name = "result_{}_{}_best_{}.json"\
                    .format(self.class_name, tr_process["preprocess"], gsvc.best_score_)

                results_dict = dict()
                results_dict["score"] = gsvc.best_score_
                results_dict["params"] = gsvc.best_params_
                results_dict["n_fold"] = self.n_fold
                results_dict["preprocessing"] = tr_process["preprocess"]
                with open(os.path.join(grid_results_path, results_best_dict_name), 'w') as grid_best_json:
                    json.dump(results_dict, grid_best_json, indent=4)

                # export parameters that the
                results_params_dict_name = "result_{}_{}_params_{}.json"\
                    .format(self.class_name, tr_process["preprocess"], gsvc.best_score_)
                with open(os.path.join(grid_results_path, results_params_dict_name), 'w') as grid_params_json:
                    json.dump(gsvc.cv_results_["params"], grid_params_json, indent=0)

                # return a list that includes the best models exported from each processing
                self.best_models_list.append(results_dict)

                print()
                print("Next train process..")
                process_counter += 1
                print()
                print()
        print(colored("Finishing training processes..", "blue"))
        print()

    def export_best_classifier(self):
        # gather best scores from the exported grid svm models
        scores = [x["score"] for x in self.best_models_list]
        print(colored("This is the max score of all the training processes: {}".format(max(scores)), "cyan"))
        for model in self.best_models_list:
            if model["score"] == max(scores):
                print("Best {} model parameters:".format(self.class_name))
                pprint(model)
                with open(os.path.join(self.exports_path, "best_model_{}".format(self.class_name)), "w") as best_model:
                    json.dump(model, best_model, indent=4)
                    print(colored("Best {} model parameters saved successfully to disk.".format(self.class_name),
                                  "cyan"))
