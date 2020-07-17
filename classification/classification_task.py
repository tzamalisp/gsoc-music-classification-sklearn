import os
from classification.classifierGRID import TrainGridClassifier
import json
from utils import load_yaml, FindCreateDirectory
from evaluation import evaluateNfold
from classification.classifierBASIC import TrainClassifier
from classification.evaluation import export_folded_instances


class ClassificationTask:
    def __init__(self, config, classifier, train_class, training_processes, X, y, exports_path, tracks):
        self.config = config
        self.classifier = classifier
        self.train_class = train_class
        self.X = X
        self.y = y
        self.training_processes = training_processes
        self.exports_path = exports_path
        self.tracks = tracks

    def run(self):
        if self.config["train_kind"] == "grid":
            grid_svm_train = TrainGridClassifier(config=self.config,
                                                 classifier=self.classifier,
                                                 class_name=self.train_class,
                                                 X=self.X,
                                                 y=self.y,
                                                 tr_processes=self.training_processes,
                                                 exports_path=self.exports_path
                                                 )
            grid_svm_train.train_grid_search_clf()
            grid_svm_train.export_best_classifier()
        elif self.classifier == "NN":
            pass

        # load best model
        best_model_name = "best_model_{}.json".format(self.train_class)
        with open(os.path.join(self.exports_path, best_model_name)) as best_model_file:
            best_model = json.load(best_model_file)

        print(best_model)

        clf_model = TrainClassifier(classifier=self.classifier, params=best_model["params"]).model()
        print("Best model loaded..")
        export_folded_instances(config=self.config, clf=clf_model,
                                n_fold=best_model["n_fold"],
                                X_array_list=self.X, y=self.y,
                                class_name=self.train_class, tracks=self.tracks,
                                process=best_model["preprocessing"],
                                exports_path=self.exports_path)

