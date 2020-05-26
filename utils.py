import os


def load_yaml():
    """
    Todo: add comments, docstring info, etc.
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


if __name__ == '__main__':
    conf_data = load_yaml()
    print(conf_data)
