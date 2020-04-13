import logging
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.DEBUG)


class DatasetExtractor:

    def __init__(self, target_directory: str):
        self.dataset = {}
        self.scaler = MinMaxScaler()
        self.target_directory = target_directory
        self.set_target_directory(target_directory)

    def set_target_directory(self, target_directory: str):
        if not os.path.exists(target_directory):
            logging.error(
                'Target directory {0} does not exist. Please reset target folder again.'.format(target_directory))
            self.target_directory = None
        else:
            pass

    def load_all_data_under_target_directory(self, ignore_list=()):
        """
        Procedure to load all the csv extension files into a dictionary under the target directory
        :param ignore_list: List of names of files which will be ignored while loading
        """
        if self.target_directory is not None:
            file_names = [dataset_file for dataset_file in os.listdir(self.target_directory) if
                          dataset_file.endswith('.csv')]
            for file_name in file_names:
                if file_name not in ignore_list:
                    self.load_specific_data(file_name)
        else:
            logging.warning('Target folder is not set yet. No data will be loaded.')

    def load_specific_data(self, file_name: str):
        """
        Procedure to load single file under the target directory
        """
        country = file_name[:file_name.index('.csv')]
        csv_content = pd.read_csv('{0}{1}'.format(self.target_directory, file_name))
        self.dataset[country] = csv_content.iloc[:, 1:2].values

    def scale_loaded_data(self):
        """
        Fits and transforms the loaded dataset
        """
        whole_dataset = np.concatenate(list(self.dataset.values()))
        self.scaler.fit(whole_dataset)
        for key, value in self.dataset.items():
            self.dataset[key] = self.scaler.transform(value)

    def get_dataset(self, day_interval=4):
        """
        Procedure which creates dataset with the given day duration
        :param day_interval: The day duration which results into output value
        """
        X, y = [], []

        # Create series of data with the given day interval
        for country, data in self.dataset.items():
            for i in range(day_interval, len(data)):
                X.append(data[i - day_interval:i, 0])
                y.append(data[i, 0])

        X, y = np.array(X), np.array(y)
        return X, y

    def scale_given_data(self, data):
        """
        Scaling method of the given data
        :param data: Data to be scaled
        """
        return self.scaler.transform(data)

    def inverse_scale_given_data(self, data):
        """
        Inverse scaling method of the given data
        :param data: Data to be scaled inversely
        """
        return self.scaler.inverse_transform(data)
