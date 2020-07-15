import os
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from transformation.transform import Transform
from sklearn.model_selection import KFold
import json

from utils import load_yaml, FindCreateDirectory, TrainingProcesses


def train_grid_search_svm(config, class_name, clf, X, y, tr_process, n_fold, exports_path):

    X_transformed = Transform(config=config,
                              df=X,
                              process=tr_process["preprocess"],
                              exports_path=exports_path
                              ).post_processing()

    # define the length of parameters
    parameters_grid = {'kernel': tr_process["kernel"],
                       'C': tr_process["C"],
                       'gamma': tr_process["gamma"],
                       'class_weight': tr_process["balanceClasses"]
                       }
    # initiate SVM classifier object
    if clf is "C-SVC":
        svm = SVC(gamma="auto", probability=True)
    else:
        raise ValueError('The classifier name must be a valid!')

    # inner CV declaration
    inner_cv = KFold(n_splits=n_fold,
                     shuffle=False,
                     )
    # initiate GridSearch Object
    gsvc = GridSearchCV(estimator=svm,
                        param_grid=parameters_grid,
                        cv=inner_cv,
                        n_jobs=config["parallel_jobs"],
                        verbose=config["grid_verbose"]
                        )

    print("Shape of X before train: {}".format(X_transformed.shape))
    print("Fitting the data to the model:")
    gsvc.fit(X_transformed, y)

    # print(gsvc.cv_results_["params"])
    print(gsvc.best_score_)
    print(gsvc.best_estimator_)
    print(gsvc.best_params_)
    print("Counted evaluations in this GridSearch Train process: {}".format(len(gsvc.cv_results_["params"])))

    exports_dir = "{}_{}".format(config.get("exports_directory"), class_name)
    exports_path = FindCreateDirectory(exports_dir).inspect_directory()

    # save best results for each train process
    grid_results_dir = os.path.join(exports_path, "results")
    grid_results_path = FindCreateDirectory(grid_results_dir).inspect_directory()
    results_best_dict_name = "result_{}_{}_{}_best.json"\
        .format(class_name, gsvc.best_score_, tr_process["preprocess"])
    results_dict = dict()
    results_dict["score"] = gsvc.best_score_
    results_dict["params"] = gsvc.best_params_
    results_dict["n_fold"] = n_fold
    with open(os.path.join(grid_results_path, results_best_dict_name), 'w') as grid_best_json:
        json.dump(results_dict, grid_best_json, indent=4)

    # export parameters that the
    results_params_dict_name = "result_{}_{}_{}_params.json"\
        .format(class_name, gsvc.best_score_, tr_process["preprocess"])
    with open(os.path.join(grid_results_path, results_params_dict_name), 'w') as grid_params_json:
        json.dump(gsvc.cv_results_["params"], grid_params_json, indent=4)