"""
This file consists of the LoggerSetup class that is used for logging.

Here, the LoggerSetup and its embedded setup_logger() method set up a new logger object with the related configurations.

    Typical usage example:

    logging_object = LoggerSetup(logger_name, logging_file_location, level_of_logging)
    logger = logging_object.setup_logger()
"""
import logging
import os
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# If log directory does not exist, create one
current_d = os.getcwd()
print(current_d)
if not os.path.exists(os.path.join(current_d, 'log')):
    os.makedirs(os.path.join(current_d, 'log'))


class LoggerSetup:
    """It sets up a logging object.

    Attributes:
        name: The name of the logger.
        log_file: The path of the logging file export.
        level: An integer that defines the logging level.
    """
    def __init__(self, name, log_file, level=1):
        """
        Inits the logger object with the corresponding parameters.

        Args:
            name (str): The name of the logger.
            log_file (str): The path the logging exports will be exported.
            level (int): The level of the logging. Defaults to 1.
        """
        self.name = name
        self.log_file = log_file
        self.level = level

# def setup_logger(name, log_file, level=logging.INFO):
    def setup_logger(self):
        """
        Function to set up as many loggers as you want. It exports the logging results to a file
        in the relevant path that is determined by the configuration file.

        :return:
        """
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setFormatter(formatter)

        logger_object = logging.getLogger(self.name)
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

        logger_object.addHandler(handler)

        return logger_object