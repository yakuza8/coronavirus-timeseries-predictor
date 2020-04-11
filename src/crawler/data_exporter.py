import csv
import logging
import os


class DataExporter:

    def __init__(self):
        self._create_directory_safely('../../resources')

    def write_dataset(self, dataset: dict):
        table_names = set()

        # Find unique table names
        for country, country_tables in dataset.items():
            for table in country_tables.keys():
                table_names.add(table)

        # Create directories for them
        for table_name in table_names:
            self._create_directory_safely('../../resources/{0}'.format(table_name))

    def _create_directory_safely(self, directory_name: str):
        try:
            os.mkdir('{0}'.format(directory_name))
            logging.debug('{0} directory is created in order to export data set.'.format(directory_name))
        except FileExistsError as e:
            logging.debug('{0} directory already exists.'.format(directory_name))
