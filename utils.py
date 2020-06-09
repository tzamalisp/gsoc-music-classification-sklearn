import os


def load_yaml():
    """
    Todo: add comments, docstring info, etc.
    :return:
    """
    try:
        import yaml
        with open(os.path.join(os.path.abspath(os.getcwd()), 'configuration.yaml')) as file:
            config_data = yaml.load(file, Loader=yaml.FullLoader)
            # print(type(config_data))
            # print(config_data)
            if isinstance(config_data, dict):
                return config_data
            else:
                return None
    except ImportError:
        print('WARNING: could not import yaml module')
        return None


class DfChecker:
    """

    """
    def __init__(self, df_check):
        """

        :param df_check:
        """
        self.df_check = df_check

    def check_df_info(self):
        """
        Prints information about the Pandas DataFrame that is generated from the relevant process.
        :return:
        """
        print('Features DataFrame head:')
        print(self.df_check.head())
        print()
        print('Information:')
        print(self.df_check.info())
        print()
        print('Shape:', self.df_check.shape)
        print('Number of columns:', len(list(self.df_check.columns)))

        if 'category' in self.df_check.columns:
            print('Track categories distribution:')
            print(self.df_check['category'].value_counts())


class FindCreateDirectory:
    """

    """
    def __init__(self, directory):
        """

        :param directory:
        """
        self.directory = directory

    def inspect_directory(self):
        """

        :return:
        """
        # find dynamically the current script directory
        path_app = os.path.join(os.path.abspath(os.getcwd()))
        full_path = os.path.join(path_app, self.directory)
        # create path directories if not exist --> else return the path
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        # print('Path {}:'.format(self.directory), full_path)
        return full_path


if __name__ == '__main__':
    conf_data = load_yaml()
    print(conf_data)

    test = FindCreateDirectory('exports').inspect_directory()
