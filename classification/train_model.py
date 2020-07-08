from classification.models import Models


class TrainModel:
    def __init__(self, config, train_data, label_data, train_class):
        self.config = config
        self.train_data = train_data
        self.label_data = label_data
        self.train_class = train_class

    def train_model(self):
        model_train = Models(config=self.config,
                             features=self.train_data,
                             labels=self.label_data,
                             class_name=self.train_class
                             )
        if self.config.get("train_kind") == "grid_svm":
            model = model_train.train_grid_search_svm()
            return model
        elif self.config.get("train_kind") == "svm":
            model = model_train.train_svm()
            return model
        elif self.config.get("train_kind") == "deep_learning":
            model = model_train.train_neural_network()
            return model
        else:
            return None
