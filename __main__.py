import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, \
    AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import torch
import numpy as np
from DataLogger import DataLogger
from EasyTorch.EasyConvNet import EasyConvNet
from EasyTorch.EasyRecNet import EasyRecNet
from EasyTorch.EasyLSTM import EasyLSTM
from EasyTorch.EasyNeuralNet import EasyNeuralNet
from Evaluate import evaluate_model
from LoadStockDataset import LoadStockDataset
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor

#increments = [2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072, 4096]
increments = [128,256,512,1024]


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
    X_train.shape[1], num_outputs=y_train.shape[1])

    # Get dummy classifiers or regressors based on the problem type
    classifiers = DummyRegressor()
    # Evaluate each model using the training and test data

    evaluate_model(classifiers, X_train, X_test, y_train, y_test, data_logger)

def sklearn_tests(dataset_id,dataset):

    # Load the dataset and split into training and testing sets

    X_train, X_test, y_train, y_test = dataset

    data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                             X_train.shape[1],num_outputs=y_train.shape[1])

    # Get sklearn classifiers or regressors based on the problem type
    classifiers = [
        LinearRegression(),
        DecisionTreeRegressor(),
        KNeighborsRegressor(),
    ]
    nums = [2,4,8,16,32,64,128]

    for i in nums:
        classifiers.append(RandomForestRegressor(n_jobs=-1,n_estimators=i),
        )
        classifiers.append(ExtraTreesRegressor(n_jobs=-1,n_estimators=i),
        )
        
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
    X_train, X_test, y_train = convert_data_to_tensors(X_train, X_test, y_train,image_bool=not True)
    data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                             X_train.shape[1],num_outputs=y_train.shape[1])


    # Evaluate each model using the training and test data
    best_score = -1
    best_model = None

    depths = []
    for i in increments:
        model = xgb.XGBRegressor(
                n_estimators=i,
                learning_rate=0.1,
                device='cuda',  # Use GPU acceleratio
                n_jobs=-1,

        )
        score = evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)
        print(f"Model Complete:{model}")
        if score > best_score:
            best_score = score
            best_model = model

    joblib.dump(best_model, 'best__xgb_model.joblib')

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
                             X_train.shape[1],num_outputs=y_train.shape[1])

    # Convert data to tensors for neural network compatibility
    X_train, X_test, y_train = convert_data_to_tensors(X_train, X_test, y_train,image_bool=not True)
    # Generate neural network models based on the problem type and output size


    best_score = -1
    best_model = None
    dropout = [0]
    lr = 0.0001
    for i in increments:
        for d in dropout:
            neurons = i
            
            h = (neurons, neurons)
            model = EasyNeuralNet(y_train.shape[1],h,d,learning_rate=lr,batch_norm=True,image_bool=False,problem_type=1,verbose=True)
            print(evaluate_model(model, X_train, X_test, y_train, y_test, data_logger))

            h = (neurons, neurons, neurons)
            model = EasyNeuralNet(y_train.shape[1],h,d,learning_rate=lr,batch_norm=True,image_bool=False,problem_type=1,verbose=True)
            print(evaluate_model(model, X_train, X_test, y_train, y_test, data_logger))

            h = (neurons, neurons, neurons, neurons)
            model = EasyNeuralNet(y_train.shape[1],h,d,learning_rate=lr,batch_norm=True,image_bool=False,problem_type=1,verbose=True)
            print(evaluate_model(model, X_train, X_test, y_train, y_test, data_logger))


    joblib.dump(best_model, 'best_NN_model.joblib')


    

def reccurent_tests(dataset_id, df,dims=2,stock_check=False):
    X_train, X_test, y_train, y_test = df

    unique_values = np.unique(y_train)
    num_unique_values = len(unique_values)

    if dims == 1:
        data_logger = DataLogger(dataset_id,  X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:],num_outputs=y_train.shape[1])
        inputs = X_train.shape[2]
    if dims == 2:
        data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:],num_outputs=y_train.shape[1])
        inputs = X_train.shape[2]

    X_train, X_test, y_train = convert_data_to_tensors(X_train, X_test, y_train, image_bool=not stock_check)

    layers = [2,3,4]
    dropouts = [0]
    for l in layers:
        for n in increments:
            for dropout in dropouts:

                model = EasyRecNet(X_train.shape[2], y_train.shape[1], n, l, dropout=dropout,criterion="HuberLoss",learning_rate=0.001, problem_type=1,
                                        verbose=True)
                evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)



def lstm_tests(dataset_id, df,dims=2,stock_check=False):
    X_train, X_test, y_train, y_test = df

    unique_values = np.unique(y_train)
    num_unique_values = len(unique_values)

    if dims == 1:
        data_logger = DataLogger(dataset_id,  X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:],num_outputs=y_train.shape[1])
        inputs = X_train.shape[2]
    if dims == 2:
        data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:],num_outputs=y_train.shape[1])
        inputs = X_train.shape[2]

    X_train, X_test, y_train = convert_data_to_tensors(X_train, X_test, y_train, image_bool=not stock_check)
    
    layers = [2,3,4]
    dropouts = [0]
    for l in layers:
        for n in increments:
            for dropout in dropouts:

                model = EasyLSTM(X_train.shape[2], y_train.shape[1], n, l, dropout,learning_rate=0.001,criterion_str="HuberLoss", problem_type=1,
                                        verbose=True)
                evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)

def cnn_tests(dataset_id, df,dims=2,stock_check=True):
    """ Test convolutional neural network (CNN) models with 3D data from a specified dataset.

    Args:
        dataset (int): Dataset identifier.
        num_outputs (int): Indicates the number of outputs for the model (classification or regression).
        dim3d (int): Specifies if the data is 3-dimensional.
    """
    #Temp, need to refactor
    problem_type = "regression"
    X_train, X_test, y_train, y_test = df
    outputs = y_train.shape[1]
    print(outputs)

    unique_values = np.unique(y_train)
    num_unique_values = len(unique_values)
    if num_unique_values == 2:
        num_unique_values = 1
    if dims == 1:
        data_logger = DataLogger(dataset_id, X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:],num_outputs=outputs)
        inputs = X_train.shape[2]
    if dims == 2:
        data_logger = DataLogger(dataset_id, X_train.shape, len(X_train.shape), X_train.shape[0],
                                 X_train.shape[1:],num_outputs=outputs)
        inputs = X_train.shape[2]

    X_train, X_test, y_train = convert_data_to_tensors(X_train, X_test, y_train, image_bool=not stock_check)
    # Generate neural network models based on the problem type and output size

    bin_prob_type = 1
    # Generate neural network models based on the problem type and output size

    hid = 2
    conv_per_blocks = [1,2,3,4]
    pool_blocks= [1,2,3,4]
    channels =  [2,4,8,16,32,64,128]

    for pool in pool_blocks:
        for conv in conv_per_blocks:
            for chan in channels:
                if problem_type == "regression":
                    model = EasyConvNet(inputs, outputs, criterion_str= "HuberLoss", dimensions=dims,
                                        num_channels=chan, conv_layers_per_block=conv,
                                        pool_blocks=pool, dense_layers=hid,problem_type=1, verbose=True)
                    evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)
                elif stock_check:
                    model = EasyConvNet(inputs, outputs, criterion_str="BCEWithLogitsLoss", dimensions=dims,
                                        num_channels=chan, conv_layers_per_block=conv,
                                        pool_blocks=pool, dense_layers=hid, verbose=True)
                    evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)
                else:
                    model = EasyConvNet(inputs, outputs, criterion_str="CrossEntropyLoss", dimensions=dims,
                                        num_channels=chan, conv_layers_per_block=conv,
                                        pool_blocks=pool, dense_layers=hid, verbose=True)

                    evaluate_model(model, X_train, X_test, y_train, y_test, data_logger)


def ensamble_tests(dataset_id,dataset):

    # Load the dataset and split into training and testing sets

    X_train, X_test, y_train, y_test = dataset

    data_logger = DataLogger(dataset_id,X_train.shape, len(X_train.shape), X_train.shape[0],
                             X_train.shape[1],num_outputs=y_train.shape[1])

    # Get sklearn classifiers or regressors based on the problem type
    regressors = [
        LinearRegression(),
        DecisionTreeRegressor(),
        KNeighborsRegressor(),
        BaggingRegressor(),
        AdaBoostRegressor()
    ]


#    for i in increments:
 #       classifiers.append((MultiOutputRegressor(BaggingRegressor(estimator=, n_estimators=i, max_samples=1/i, max_features=1/i)),))

        
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

if __name__ == '__main__':
    id = "Experiment 1 Default Data"
    
    #sk 0
    #xbg 1
    #nn 2
    #rnn 3
    #rnn nerf -3
    #lstm 4
    #lstm nerf -4
    #cnn 5
    #cnn nerf -5
    #cnn IGTD 6
    #bagging 7
    conds = [2,3,4]
    days_obs = 1
    #conds = [1]
    for cond in conds:
        if cond == 0:
            ld = LoadStockDataset(dataset_index=1,normalize=1)
            dataset = ld.get_train_test_split()
            dummy_tests(id,dataset)
            sklearn_tests(id,dataset)
        if cond == 1:
            ld = LoadStockDataset(dataset_index=1,normalize=1)
            dataset = ld.get_train_test_split()
            xgb_tests(id,dataset)
        if cond == 2:
            ld = LoadStockDataset(dataset_index=1,normalize=1)
            dataset = ld.get_train_test_split()
            nn_tests(id,dataset)
        if cond == -3:
            ld = LoadStockDataset(dataset_index=1,normalize=1)
            dataset = ld.get_3d(version=0)
            reccurent_tests(id,dataset)
        if cond == 3:
            ld = LoadStockDataset(dataset_index=days_obs,normalize=1)
            dataset = ld.get_3d(version=1)
            reccurent_tests(id,dataset)
        if cond == 4:
            ld = LoadStockDataset(dataset_index=days_obs,normalize=1)
            dataset = ld.get_3d(version=1)        
            lstm_tests(id,dataset)
        if cond == -4:
            ld = LoadStockDataset(dataset_index=1,normalize=1)
            dataset = ld.get_3d(version=0)        
            lstm_tests(id,dataset)
        if cond == -5:
            ld = LoadStockDataset(dataset_index=1,normalize=1)
            dataset = ld.get_3d(version=0)
            cnn_tests(id,dataset,dims=1)
        if cond == 5:
            ld = LoadStockDataset(dataset_index=days_obs,normalize=1)
            dataset = ld.get_3d(version=1)
            cnn_tests(id,dataset,dims=1)
        if cond == 6:
            ld = LoadStockDataset(dataset_index=days_obs,normalize=1)
            dataset = ld.get_3d(version=2)
            cnn_tests(id,dataset,dims=2)