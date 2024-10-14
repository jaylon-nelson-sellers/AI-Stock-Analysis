from typing import Union, Callable
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.dataset import ValidSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.callbacks import Callback, LRScheduler, EarlyStopping



def create_torch_model(model: Callable, problem_type: int, criterion_str="MSELoss", batch_size: int = 64,
                       learning_rate: float = 0.001,
                       max_iter: int = 10000, early_stopping: bool = True, verbose: bool = False,
                       validation_split: int = 10,
                       log: bool = False, ) -> Union[NeuralNetClassifier, NeuralNetRegressor]:
    """
    Creates a neural network for classification or regression using Skorch.

    :param model_class: The neural network model class
    :param problem_type: 'classification' or 'regression'
    :param output_size: Number of output neurons
    :param hidden_layer_sizes: Tuple specifying sizes of hidden layers
    :param dropout_prob: Dropout probability
    :param batch_norm: Whether to use batch normalization
    :param batch_size: Size of each batch during training
    :param learning_rate: Learning rate for the optimizer
    :param max_iter: Maximum number of epochs
    :param early_stopping: Enable early stopping
    :param verbose: Verbosity level
    :param validation_split: Fraction of data to be used for validation
    :param log: Enable logging of important information
    :return: Configured neural network
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = NeuralNetRegressor if problem_type else NeuralNetClassifier
    if criterion_str == "HuberLoss":
        criterion = nn.HuberLoss()
    if criterion_str == "MSELoss":
        criterion = nn.MSELoss()
    elif criterion_str == "BCELoss":
        criterion = nn.BCELoss
    elif criterion_str == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss
    elif criterion_str == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss

    if problem_type:
        if early_stopping:
            callbacks = [LRScheduler(policy=ReduceLROnPlateau, monitor='valid_loss', mode='min', patience=5, factor=0.1,
                                     threshold_mode='rel',verbose=True),
                         EarlyStopping(monitor='valid_loss', patience=8, threshold=0.0001, threshold_mode='rel',
                                       lower_is_better=True)]
    else:
        if early_stopping:
            callbacks = [CombinedLRScheduler(10), CombinedEarlyStopping(15)]

    network = net(model,
                  criterion=criterion,
                  optimizer=torch.optim.Adam,
                  max_epochs=max_iter,
                  train_split=ValidSplit(validation_split, stratified=False),
                  lr=learning_rate,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  device=device,
                  verbose=verbose,
                  )

    if log:
        print(f"Model created for {problem_type} with device {device}")

    return network


def compute_hidden_size(input_size: int, output_size: int, method: int = 0) -> int:
    """
    Computes a suggested hidden layer size based on input size and method.

    :param input_size: Size of the input layer
    :param output_size: Size of the output layer
    :param method: Method to calculate hidden size (0-3)
    :return: Suggested hidden layer size
    """
    methods = {
        0: (input_size + output_size) / 2,
        1: input_size,
        2: input_size * 1.5,
        3: input_size * 2
    }
    hidden_size = methods.get(method, (input_size + output_size) / 2)
    return int(hidden_size)


class CombinedLRScheduler(Callback):
    def __init__(self, patience=10, monitor_acc='valid_acc', monitor_loss='valid_loss', factor=0.1, min_lr=1e-7,
                 threshold_acc=0.001, threshold_loss=0.001, lower_is_better_acc=False, lower_is_better_loss=True):
        self.patience = patience
        self.monitor_acc = monitor_acc
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.min_lr = min_lr
        self.lower_is_better_acc = lower_is_better_acc
        self.lower_is_better_loss = lower_is_better_loss
        self.threshold_acc = threshold_acc
        self.threshold_loss = threshold_loss
        self.best_acc = None
        self.best_loss = None
        self.wait = 0

    def on_epoch_end(self, net, **kwargs):
        current_acc = net.history[-1][self.monitor_acc]
        current_loss = net.history[-1][self.monitor_loss]

        if self.best_acc is None or self.best_loss is None:
            self.best_acc = current_acc
            self.best_loss = current_loss
            return
        if self.lower_is_better_acc:
            acc_improved = (current_acc < self.best_acc + self.threshold_acc)
        else:
            acc_improved = (current_acc > self.best_acc + self.threshold_acc)
        loss_improved = (current_loss < self.best_loss - self.threshold_loss)

        if acc_improved or loss_improved:
            self.best_acc = current_acc
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                for param_group in net.optimizer_.param_groups:
                    new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                # print(f'Reducing learning rate to {new_lr}')


class CombinedEarlyStopping(Callback):
    def __init__(self, patience=20, monitor_acc='valid_acc', monitor_loss='valid_loss', threshold_acc=0.001,
                 threshold_loss=0.001, lower_is_better_acc=False, lower_is_better_loss=True):
        self.patience = patience
        self.monitor_acc = monitor_acc
        self.monitor_loss = monitor_loss
        self.threshold_acc = threshold_acc
        self.threshold_loss = threshold_loss
        self.lower_is_better_acc = lower_is_better_acc
        self.lower_is_better_loss = lower_is_better_loss
        self.best_acc = None
        self.best_loss = None
        self.wait = 0

    def on_epoch_end(self, net, **kwargs):
        current_acc = net.history[-1][self.monitor_acc]
        current_loss = net.history[-1][self.monitor_loss]

        if self.best_acc is None or self.best_loss is None:
            self.best_acc = current_acc
            self.best_loss = current_loss
            return

        if self.lower_is_better_acc:
            acc_improved = (current_acc < self.best_acc + self.threshold_acc)
        else:
            acc_improved = (current_acc > self.best_acc + self.threshold_acc)
        loss_improved = (current_loss < self.best_loss - self.threshold_loss)
        if acc_improved or loss_improved:
            self.best_acc = current_acc
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'Early stopping after {self.patience} epochs without improvement in both accuracy and loss')
                raise KeyboardInterrupt
