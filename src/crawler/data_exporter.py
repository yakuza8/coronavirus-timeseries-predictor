import csv
import logging
import os


class DataExporter:
    """
    Dataset creator in CSV format where data set is in form of the following structure

    Format:
    {
        'Red Country': {
            'Orange Table': {
               'x': ['Day1', 'Day2', ..., 'Day8'],
               'y': [8, 7, ..., 1]
            },
            ...
        },
        'Pink Country': {
            ...
        },
        ...
    }
    """

    RESOURCE_FOLDER_PATH = '../../resources'
    CSV_HEADERS = ['Date', 'Value']

    def __init__(self):
        self._create_directory_safely(self.RESOURCE_FOLDER_PATH)

    def write_dataset(self, dataset: dict):
        logging.info('Data export will start.')
        table_names = set()

        # Find unique table names
        for country, country_tables in dataset.items():
            for table in country_tables.keys():
                table_names.add(table)

        # Create directories for them
        for table_name in table_names:
            self._create_directory_safely('{0}/{1}'.format(self.RESOURCE_FOLDER_PATH, table_name))

        # Create files
        for country, country_tables in dataset.items():
            logging.debug('Country {0} data will be written.'.format(country))
            for table_name, table_content in country_tables.items():
                self._write_table_content(country, table_name, table_content)
        logging.info('Data export finished as CSV format.')

    @staticmethod
    def _create_directory_safely(directory_name: str):
        try:
            os.mkdir('{0}'.format(directory_name))
            logging.debug('{0} directory is created in order to export data set.'.format(directory_name))
        except FileExistsError:
            logging.debug('{0} directory already exists.'.format(directory_name))

    def _write_table_content(self, country_name: str, table_name: str, table_content: dict):
        logging.debug('Country {0}\'s {1} data will be written.'.format(country_name, table_name))
        with open(
                '{0}/{1}/{2}.csv'.format(self.RESOURCE_FOLDER_PATH, table_name, '_'.join(country_name.lower().split())),
                'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADERS)
            writer.writerows([(x, y) for x, y in zip(table_content['x'], table_content['y']) if y is not None and y != 0])
        logging.debug('Country {0}\'s {1} data export finished.'.format(country_name, table_name))
