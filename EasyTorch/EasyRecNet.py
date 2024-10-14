import torch
import torch.nn as nn
from EasyTorch.EasyTorch import create_torch_model
from typing import Any

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RecurrentNet(nn.Module):
    """
    A customizable recurrent neural network (RNN) class.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_prob: float = 0.0):
        """
        Initializes the recurrent neural network with the given parameters.

        :param input_size: Size of the input features.
        :param output_size: Size of the output layer.
        :param hidden_size: Number of features in the hidden state.
        :param num_layers: Number of recurrent layers.
        :param dropout_prob: Dropout probability.
        """
        super(RecurrentNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu',
                          dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        :param x: Input tensor.
        :return: Output tensor.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class EasyRecNet:
    """
    A class to simplify recurrent neural network creation and training using Torch.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_prob: float = 0.0,
                 criterion: str = "MSELoss", problem_type: int = 0, batch_size: int = 64, learning_rate: float = 0.001,
                 max_iter: int = 10000, early_stopping: bool = True, verbose: bool = False,
                 num_splits: int = 10,
                 log: bool = False):
        """
        Initializes the EasyRecNet with the given parameters.

        :param input_size: Size of the input features.
        :param output_size: Size of the output layer.
        :param hidden_size: Number of features in the hidden state.
        :param num_layers: Number of recurrent layers.
        :param dropout_prob: Dropout probability.
        :param criterion: Loss function.
        :param problem_type: Type of problem (e.g., regression, classification).
        :param batch_size: Batch size for training.
        :param learning_rate: Learning rate for the optimizer.
        :param max_iter: Maximum number of iterations.
        :param early_stopping: Whether to use early stopping.
        :param verbose: Whether to print verbose logs.
        :param num_splits (int, optional): K-Fold Split of the data. Defaults to 10
        :param log: Whether to log training progress.
        """
        self.model = RecurrentNet(input_size, output_size, hidden_size, num_layers, dropout_prob)
        self.network = create_torch_model(self.model, problem_type, criterion, batch_size, learning_rate, max_iter,
                                          early_stopping, verbose, num_splits, log)
        self.log = log

    def fit(self,  X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fits the model to the given data.

        :param X: Input data.
        :param y: Target labels.
        :return: Trained model.
        """
        if self.log:
            print("Training started...")
        result = self.network.fit(X, y)
        if self.log:
            print("Training completed.")
        return result

    def predict(self, X: torch.Tensor,) -> torch.Tensor:
        """
        Predicts the output for the given input data.

        :param X: Input data.
        :return: Predicted output.
        """
        if self.log:
            print("Prediction started...")
        result = self.network.predict(X)
        if self.log:
            print("Prediction completed.")
        return result

    def __str__(self) -> str:
        """
        String representation of the EasyRecNet.

        :return: String describing the network configuration.
        """
        return f"EasyRecNet(Layers: {self.model.num_layers}, Neurons per Layer: {self.model.hidden_size}, Dropout: {self.model.dropout_prob})"
