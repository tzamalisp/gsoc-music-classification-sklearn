from classification.models import Models


class TrainModel:
    def __init__(self, config, train_data, label_data, train_class):
        self.config = config
        self.train_data = train_data
        self.label_data = label_data
        self.train_class = train_class

    def train_model(self):
        if self.config.get("train_kind") == "grid_svm":
            model_trained = Models(config=self.config,
                                   features=self.train_data,
                                   labels=self.label_data,
                                   class_name=self.train_class
                                   ).train_grid_search()
            return model_trained
        elif self.config.get("train_kind") == "svm":
            model_trained = Models(config=self.config,
                                   features=self.train_data,
                                   labels=self.label_data,
                                   class_name=self.train_class
                                   ).train_svm()
            return model_trained
        elif self.config.get("train_kind") == "deep_learning":
            model_trained = Models(config=self.config,
                                   features=self.train_data,
                                   labels=self.label_data,
                                   class_name=self.train_class
                                   ).train_neural_network()

            return model_trained
        else:
            return None
