from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from src.predictor.rnn_predictor import build_model, get_dataset_split

# Get dataset
resource_folder = '../../resources/'
total_death_folder = 'Total Deaths/'
_, (X_train, X_test, y_train, y_test) = get_dataset_split(resource_folder, total_death_folder)

model = KerasRegressor(build_fn=build_model)
batch_size = [10, 20, 30]
epochs = [50, 100]
layer1 = [30, 40, 50]
layer2 = [30, 40, 50]
layer3 = [30, 40, 50]
dropout = [0.15, 0.20, 0.25]
optimizer = ['SGD', 'Adagrad', 'Adam', 'Adamax']

# Here we need to give input dimension parameter so that layers can be compiled
param_grid = dict(input_dimension=[X_train.shape[1]], batch_size=batch_size, epochs=epochs,
                  layer1=layer1, layer2=layer2, layer3=layer3,
                  dropout=dropout, optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)
