"""
This file consists of the LoggerSetup class that is used for logging.

Here, the LoggerSetup and its embedded setup_logger() method set up a new logger object with the related configurations.

    Typical usage example:

    logging_object = LoggerSetup(logger_name, logging_file_location, level_of_logging)
    logger = logging_object.setup_logger()
"""
import logging
import os
from utils import load_yaml, FindCreateDirectory

# # load yaml configuration file to a dict
# config_data = load_yaml()
# # If log directory does not exist, create one
# current_d = os.getcwd()
# if config_data["log_directory"] is None or config_data["log_directory"] is None:
#     if not os.path.exists(os.path.join(current_d, "logs_dir")):
#         os.makedirs(os.path.join(current_d, "logs_dir"))
#         log_path = os.path.join(current_d, "logs_dir")
# else:
#     log_path = FindCreateDirectory(config_data["log_directory"]).inspect_directory()


class LoggerSetup:
    """It sets up a logging object.

    Attributes:
        name: The name of the logger.
        log_file: The path of the logging file export.
        level: An integer that defines the logging level.
    """
    def __init__(self, config, exports_path, name, train_class, mode, level=1):
        """
        Inits the logger object with the corresponding parameters.

        Args:
            name (str): The name of the logger.
            log_file (str): The path the logging exports will be exported.
            level (int): The level of the logging. Defaults to 1.
        """
        self.config = config
        self.exports_path = exports_path
        self.name = name
        self.train_class = train_class
        self.mode = mode
        self.level = level

        self.exports_dir = ""
        self.logs_path = ""

    def setup_logger(self):
        """
        Function to set up as many loggers as you want. It exports the logging results to a file
        in the relevant path that is determined by the configuration file.

        :return:
        """
        self.exports_dir = "{}_{}".format(self.config.get("exports_directory"), self.train_class)
        self.logs_path = FindCreateDirectory(self.exports_path,
                                             os.path.join(self.exports_dir, "logs")).inspect_directory()

        # Create a custom logger
        logger_object = logging.getLogger(self.name)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(self.logs_path, "{}.log".format(self.name)), mode=self.mode)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger_object.addHandler(c_handler)
        logger_object.addHandler(f_handler)

        if self.level is None:
            logger_object.setLevel(logging.INFO)
        elif self.level is 0:
            logger_object.setLevel(logging.DEBUG)
        elif self.level is 1:
            logger_object.setLevel(logging.INFO)
        elif self.level is 2:
            logger_object.setLevel(logging.WARNING)
        elif self.level is 3:
            logger_object.setLevel(logging.ERROR)
        elif self.level is 4:
            logger_object.setLevel(logging.CRITICAL)
        else:
            print('Please define correct one of the Debug Levels:\n'
                  '0: DEBUG\n'
                  '1: INFO\n'
                  '2: WARNING\n'
                  '3: ERROR\n'
                  '4: CRITICAL')

        return logger_object
