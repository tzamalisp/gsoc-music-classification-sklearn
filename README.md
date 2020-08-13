# Machine Learning Infrastructure with scikit-learn (GSoC 2020)

This repository contains the tool that is built for training SVM models of 
AcousticBrainz's datasets, as well as predicting where a single AcousticBrainz 
track instance can be classified based on the trained models. It is part of the 
*Google Summer of Code 2020* in collaboration with the **MetaBrainz** Open-Source 
organization.

Given a dataset, a Grid Search algorithm using n-fold cross-validation is executed 
for an exhaustive search over specified parameter values for an estimator.

A final model is trained with all the data (without a validation set) featuring 
the best parameter combination in terms of accuracy.

Finally, a prediction functionality is part of the tool, which gives the user the 
capability of predicting where a track instance is classified based on a trained model.


## Functionalities

### Train
The main model training method is the `create_classification_project` which is located in
the `create_classification_project.py` Python script. It can be imported as a module or 
run as an executable too. It requires a path to a folder that contains subfolders 
composed of the groundtruth yaml file/s (tracks, tracks paths, labels, target class), and
the features (low-level data) in JSON format.

```
$ python create_classification_project.py --help
usage: create_classification_project.py [-h] [--groundtruth] [--file] [--exportsdir] [--path]
                                        [--seed logging] [--seed] [--jobs] [--verbose]

Generates a model trained using descriptor files specified in the groundtruth yaml file.

positional arguments:
  -g, --groundtruth      Path of the main dataset directory containing the 
                        groundtruth yaml file/s. (required)

  -f, --file            Name of the project configuration file (.yaml) will be stored. 
                        If not specified it takes automatically the name <project_CLASS_NAME>."

  -d, --exportsdir      Name of the exports directory that the project's results 
                        will be stored (best model, grid models, transformation 
                        pipelines, folded and shuffled dataset).

  -p --path             Path where the project results will be stored. If empty,
                        the results will be saved in the main app directory.

optional arguments:
  -h, --help            show the help message about the arguments and exit

  -l, --logging         The logging level (int) that will be printed (0: DEBUG, 1: INFO, 
                        2: WARNING, 3: ERROR, 4: CRITICAL). Can be set only in the
                        prescribed integer values (0, 1, 2, 3, 4)

  -s, --seed            Seed (int) is used to generate the random shuffled dataset 
                        applied later to folding. If no seed is specified, the seed
                        will be automatically set to current clock value.

  -j, --jobs            Parallel jobs (int). Set a value of cores to be used.
                        The default is -1, which means that all the available cores
                        will be used.
  
  -v, --verbose         Controls the verbosity (int) of the Grid Search print messages
                        on the console: the higher, the more messages.
```

For example, a path directory structure could be like this one:

    dataset (e.g. danceability)
    |- features
    |  |-happy
    |  |  |- 1.json
    |  |  |- 2.json
    |  |  |- 3.json
    |  |  |- 4.json
    |  |-sad
    |  |  |- 1.json
    |  |  |- 2.json
    |  |  |- 3.json
    |- metadata
    |  |- groundtruth.yaml
    
The tool will train a model with 2 classes (happy, sad), with 4 and 3 files in each class, respectively.

The tool generates a `.yaml` project file to the path and exports directory specified or by the 
arguments or automatically by the tool itself. This project file contains information about the 
preprocessing steps that are followed through the training process, as well as the path and directory
where the results after the model training will be stored to. These results contain:

* Shuffled tracks in CSV format.
* Logs.
* Best models resulted from eah training process (in `.pkl` files).
* Features transformation pipelines (in `.pkl` files) followed in each training process 
(e.g. basic, normalization, gaussianization, etc.).
* Best resulted models' parameters and all the parameters that were executed in each training
process.
* Classification reports and confusion matrices from the evaluation, both in the k-fold 
cross-validation evaluation method and to whole dataset.
* Image with the accuracies distribution in each fold.
* Best model's score, parameters, preprocess, fold that were followed in training. The result is
in JSON format.
* Best model saved in `.pkl` format after the fitting to the whole dataset.

### Train Process



### Predict



```
$ python predict.py --help
usage: predict.py [-h] [--path] [--file] [--track] [--logging]

positional arguments:
  -p --path             Path where the project file (.yaml) is stored if not in the 
                        same file where the app is.

  -f, --file            Name of the project configuration file (.yaml) that 
                        is to be loaded. (required)

  -t --track            MBID of the the low-level data from the AcousticBrainz API.
                        (required)

optional arguments:
  -h, --help            show the help message about the arguments and exit

  -l, --logging         The logging level (int) that will be printed (0: DEBUG, 1: INFO, 
                        2: WARNING, 3: ERROR, 4: CRITICAL). Can be set only in the
                        prescribed integer values (0, 1, 2, 3, 4)
```
