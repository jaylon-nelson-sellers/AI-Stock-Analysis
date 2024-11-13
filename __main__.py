import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, \
    AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import torch
import numpy as np
from DataLogger import DataLogger
from EasyTorch.EasyLSTM import EasyLSTM
from EasyTorch.EasyNeuralNet import EasyNeuralNet
from Evaluate import evaluate_model
from LoadStockDataset import LoadStockDataset
import xgboost as xgb

def convert_data_to_tensors(X_train, X_test, y_train,image_bool=False):
    """ Convert numpy arrays to PyTorch tensors.

    Args:
        X_train (np.array): Training features.
        X_test (np.array): Testing features.
        y_train (np.array): Training labels.

    Returns:
        tuple: Tuple containing tensors for training features, testing features, and training labels.
    """
    # Check device availability (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Check if the data is a pandas DataFrame and convert to NumPy if true

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.to_numpy()

    if isinstance(X_train, pd.Series):
        X_train = X_train.to_numpy()
    if isinstance(X_test, pd.Series):
        X_test = X_test.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    # Ensure data is numpy array before converting to tensor
    if isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray) and isinstance(y_train, np.ndarray):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        if image_bool:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        return X_train_tensor, X_test_tensor, y_train_tensor
    else:
        raise TypeError("Input data should be either NumPy arrays or pandas DataFrames.")

def dummy_tests(dataset_id,dataset):
    # Load the dataset and split into training and testing sets
    X_train, X_test, y_train, y_test = dataset
    # Initialize the data logger to record dataset details
    data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape),  X_train.shape[0],
    X_train.shape[1])

    # Get dummy classifiers or regressors based on the problem type
    classifiers = DummyRegressor()
    # Evaluate each model using the training and test data

    evaluate_model(classifiers, X_train, X_test, y_train, y_test, data_logger)

def sklearn_tests(dataset_id,dataset):

    # Load the dataset and split into training and testing sets

    X_train, X_test, y_train, y_test = dataset

    data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                             X_train.shape[1])

    # Get sklearn classifiers or regressors based on the problem type
    classifiers = [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(n_jobs=-1),
        ExtraTreesRegressor(n_jobs=-1)
    ]
    # Evaluate each model using the training and test data
    best_score = -1
    best_model = None
    for model in classifiers:
        score = evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)
        print(f"Model Complete:{model}")
        if score > best_score:
            best_score = score
            best_model = model

    joblib.dump(best_model, 'best_model.joblib')

def xgb_tests(dataset_id,dataset):

    # Load the dataset and split into training and testing sets

    X_train, X_test, y_train, y_test = dataset

    data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                             X_train.shape[1])


    # Evaluate each model using the training and test data
    best_score = -1
    best_model = None
    nums = [2,4,8,16,32,64,128,256,512,1024]
    for n in nums:
        model = xgb.XGBRegressor(
                n_estimators=n,
                learning_rate=0.1,
                device='cuda',  # Use GPU acceleratio
                n_jobs=-1,

            )
        score = evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)
        print(f"Model Complete:{model}")
        if score > best_score:
            best_score = score
            best_model = model

    joblib.dump(best_model, 'best_model.joblib')

def nn_tests(dataset_id, dataset):
    """ Test neural network models on a specified dataset.

    This method loads the dataset, prepares tensors for neural network training, and evaluates various models.

    Args:
        dataset (int): Dataset identifier.
        problem_type (int): Indicates classification (0) or regression (1).
    """
    X_train, X_test, y_train, y_test = dataset

    unique_values = np.unique(y_train)
    num_unique_values = len(unique_values)
    if num_unique_values == 2:
        num_unique_values = 1

    data_logger = DataLogger(dataset_id, X_train.shape, len(X_train.shape), X_train.shape[0],
                             X_train.shape[1])

    # Convert data to tensors for neural network compatibility
    X_train, X_test, y_train = convert_data_to_tensors(X_train, X_test, y_train,image_bool=not True)
    # Generate neural network models based on the problem type and output size

    #change this to represent y


    models = create_nn_models(y_train.shape[1],10,stock_check=True)

    # Evaluate each model using the prepared data

    best_score = -1
    best_model = None
    for model in models:
        score = evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)
        if score > best_score:
            best_score = score
            best_model = model

    joblib.dump(best_model, 'best_NN_model.joblib')
def create_nn_models(problem_type: str, output_size: int,num_layers=2,image_check=False,stock_check=False):
    """ Generate neural network models with varying hidden layer configurations.

    Args:
        problem_type (str): Indicates the type of problem (classification or regression).
        output_size (int): The size of the output layer, typically the number of classes or regression outputs.

    Returns:
        list: A list of neural network models with different hidden layer sizes for testing.
    """
    # Define a range of hidden layer sizes

    hiddens = []
    iters = 15
    nn = 2
    for i in range(iters):
        neurons = nn * (2 ** i)
        hiddens.append((neurons, neurons))
    dropout = 0

    return [EasyNeuralNet(output_size,h,dropout,batch_norm=True,learning_rate=.001,image_bool=False,problem_type=1,verbose=True)
            for h in hiddens]


def lstm_tests(dataset_id, df,dims=2,stock_check=False):
    X_train, X_test, y_train, y_test = df

    unique_values = np.unique(y_train)
    num_unique_values = len(unique_values)

    if dims == 1:
        data_logger = DataLogger(dataset_id,  X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:])
        inputs = X_train.shape[2]
    if dims == 2:
        data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:])
        inputs = X_train.shape[2]

    X_train, X_test, y_train = convert_data_to_tensors(X_train, X_test, y_train, image_bool=not stock_check)
    neuron_sizes = [2,4,8,16,32,64,128,256,512,1024]
    layers = [2,3,4,5]
    dropouts = [0.5]
    for l in layers:
        for n in neuron_sizes:
            for dropout in dropouts:

                model = EasyLSTM(X_train.shape[2], y_train.shape[1], n, l, dropout,criterion_str="HuberLoss", problem_type=1,
                                        verbose=True)
                evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)


if __name__ == '__main__':
    id = "Test 2 batch"
    twod = False
    ld = LoadStockDataset(dataset_index=1,normalize=1)

    if twod:
        dataset = ld.get_3d()
        lstm_tests(id,dataset)
    else:
        dataset = ld.get_train_test_split()
        #dummy_tests(id,dataset)
        sklearn_tests(id,dataset)
        #xgb_tests(id,dataset)
        #nn_tests(id,dataset)