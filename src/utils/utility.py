import os
import yaml
import joblib
from typing import Dict
from datetime import datetime
from keras.models import load_model


class Utility:
    def __init__(self):
        self.current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # set device (gpu ro cpu)
    @staticmethod
    def set_device_option(device):
        if device.split(':')[0] == 'gpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load training hyperparameters
    @staticmethod
    def load_params(yaml_file: str) -> Dict:
        with open(yaml_file, 'r') as stream:
            params = yaml.safe_load(stream)
        return params

    def save_model(self, model_in, directory, model_type):
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = directory + "/" + model_type + "_" + self.current_datetime + ".hdf5"
        model_in.save(file_path)

    @staticmethod
    def load_model(directory, model_name):
        file_path = directory + "/" + model_name

        return load_model(file_path, compile=False)

    def save_pickle(self, best_model, directory, model_name):
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = directory + "/" + model_name + "_" + self.current_datetime + ".pkl"
        joblib.dump(best_model, file_path)

    @staticmethod
    def load_pickle(directory, model_name):
        file_path = directory + "/" + model_name + ".pkl"
        joblib.load(file_path)
