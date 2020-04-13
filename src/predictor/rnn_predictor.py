import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from src.predictor.dataset_extractor import DatasetExtractor

resource_folder = '../../resources/'
total_death_folder = 'Total Deaths/'


def get_dataset_split(main_folder: str, target_data_folder: str):
    # Get dataset into memory
    _target_directory = '{0}{1}'.format(main_folder, target_data_folder)

    data_extractor = DatasetExtractor(_target_directory)
    data_extractor.load_all_data_under_target_directory()
    data_extractor.scale_loaded_data()

    # Get dataset and reshape it
    X, y = data_extractor.get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    return data_extractor, (X_train, X_test, y_train, y_test)


def build_model(input_dimension: int, optimizer='Adam', layer1=50, layer2=50, layer3=40, dropout=0.20):
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


# Get dataset and create model
data_extractor, (X_train, X_test, y_train, y_test) = get_dataset_split(resource_folder, total_death_folder)
model = build_model(X_train.shape[1])
history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.2)
# Evaluate model and get metrics
train_evaluation = model.evaluate(X_train, y_train, verbose=0)
test_evaluation = model.evaluate(X_test, y_test, verbose=0)
