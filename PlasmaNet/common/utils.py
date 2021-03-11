import os
import pickle


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_obj(obj, name):
    """ Save obj using pickle """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """ Load object using pickle """
    with open(name, 'rb') as f:
        return pickle.load(f)
