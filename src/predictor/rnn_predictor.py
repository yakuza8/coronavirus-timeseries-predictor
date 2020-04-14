import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from src.predictor.dataset_extractor import DatasetExtractor

resource_folder = '../../resources/'
total_death_folder = 'Total Deaths/'


def get_dataset_split(main_folder: str, target_data_folder: str, day_interval: int = 4):
    """
    :param main_folder: Main folder of resources
    :param target_data_folder: Target folder where the dataset information is obtained like Total Deaths
    :param day_interval: Date interval where data will be given with that interval to RNN
    :return: extractor, (*dataset)
    """
    # Get dataset into memory
    _target_directory = '{0}{1}'.format(main_folder, target_data_folder)

    data_extractor = DatasetExtractor(_target_directory)
    data_extractor.load_all_data_under_target_directory()
    data_extractor.scale_loaded_data()

    # Get dataset and reshape it
    X, y = data_extractor.get_dataset(day_interval=day_interval)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    return data_extractor, (X_train, X_test, y_train, y_test)


def build_model(input_dimension: int, optimizer='Adam', layer1=70, layer2=60, layer3=50, dropout=0.20) -> Sequential:
    """
    Model building functionality
    """
    # Create model as RNN with stacked LSTM layers
    predictor = Sequential()
    # First LSTM layer
    predictor.add(LSTM(units=layer1, return_sequences=True, input_shape=(input_dimension, 1)))
    predictor.add(Dropout(dropout))
    # Second LSTM layer
    predictor.add(LSTM(units=layer2, return_sequences=True))
    predictor.add(Dropout(dropout))
    # Last LSTM layer
    predictor.add(LSTM(units=layer3))
    predictor.add(Dropout(dropout))
    # Final output
    predictor.add(Dense(units=1))
    # Compile model
    predictor.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae', 'mape', 'cosine'])
    return predictor


def make_prediction_with_model(_model: Sequential, _data_extractor: DatasetExtractor, _data: np.ndarray) -> int:
    """
    Predictor function over the model by applying data transformations on the givven data with the given extractor
    :param _model: Model which will perform prediction
    :param _data_extractor: Data processor which will scale input and inverse scale output
    :param _data: Numpy vector which is expected to be length of day_interval i.e. shape of (day_interval,)
    :return: Numeric value
    """
    scaled_data = data_extractor.scale_given_data(_data.reshape(-1, 1, ))
    prediction = model.predict(scaled_data.reshape(1, -1, 1))
    return data_extractor.inverse_scale_given_data(prediction)


def dump_model_and_extractor(_model: Sequential, _data_extractor: DatasetExtractor, main_folder: str, name: str) -> None:
    """
    Dump model under the given main folder
    """
    import pickle
    with open('{0}{1}'.format(main_folder, name), 'wb') as f:
        pickle.dump(_model, f)
        pickle.dump(_data_extractor, f)


def load_model_and_extractor(main_folder: str, name: str):
    """
    Load model from the given main folder
    """
    import pickle
    with open('{0}{1}'.format(main_folder, name), 'rb') as f:
        _model = pickle.load(f)
        _data_extractor = pickle.load(f)
        return _model, _data_extractor


if __name__ == '__main__':
    should_built = False

    if should_built:
        # Get dataset and create model
        day_interval_for_rnn = 4
        data_extractor, (X_train, X_test, y_train, y_test) = get_dataset_split(resource_folder, total_death_folder,
                                                                               day_interval=day_interval_for_rnn)
        model = build_model(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=10, batch_size=20, validation_split=0.2)
        # Evaluate model and get metrics
        train_evaluation = model.evaluate(X_train, y_train, verbose=0)
        test_evaluation = model.evaluate(X_test, y_test, verbose=0)

        result = make_prediction_with_model(model, data_extractor, np.array([900, 1000, 1100, 1200]))

        # Dump model
        dump_model_and_extractor(model, data_extractor, resource_folder, 'corona_rnn_model')
    else:
        # Load model
        model, data_extractor = load_model_and_extractor(resource_folder, 'corona_rnn_model')
        result = make_prediction_with_model(model, data_extractor, np.array([900, 1000, 1100, 1200]))

