import torch
import torch.nn as nn
from EasyTorch.EasyTorch import create_torch_model

# Determine if CUDA (GPU) is available for model training, else use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTMNet(nn.Module):
    """
       LSTMNet Class: Defines the architecture of an LSTM network.

       Attributes:
           hidden_size (int): Number of neurons in each LSTM layer.
           num_layers (int): Number of stacked LSTM layers.
           lstm (nn.LSTM): LSTM layer.
           fc (nn.Linear): Fully connected layer for final output.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_prob: float = 0.5):
        """
        Initializes the LSTM layer and the fully connected layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_size (int): Number of neurons in each LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            dropout_prob (float): Dropout probability for regularization (default is 0.0).
        """
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.LazyLinear(output_size)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
       Defines the forward pass of the network.

       Args:
           x (torch.Tensor): Input tensor.

       Returns:
           torch.Tensor: Output tensor after passing through LSTM and fully connected layer.
       """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


class EasyLSTM:
    """
    EasyLSTM Class: Simplifies the creation, training, and prediction of the LSTM model using EasyTorch.

    Attributes:
        model (torch.nn.Module): The LSTM model created using EasyTorch.
        hidden_size (int): Number of neurons in each LSTM layer.
        num_layers (int): Number of stacked LSTM layers.
        dropout_prob (float): Dropout probability for regularization.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_prob: float = 0.0,
                 criterion_str: str = "MSELoss", problem_type: int = 0, batch_size: int = 64,
                 learning_rate: float = 0.001,
                 max_iter: int = 10000, early_stopping: bool = True, verbose: bool = False,
                 num_splits: int = 10,
                 log: bool = False):
        """
       Initializes the EasyLSTM instance with various hyperparameters and model configurations.

       Args:
           input_size (int): Number of input features.
           output_size (int): Number of output features.
           hidden_size (int): Number of neurons in each LSTM layer.
           num_layers (int): Number of stacked LSTM layers.
           dropout_prob (float): Dropout probability for regularization.
           criterion_str (str): Loss function (e.g., "MSELoss").
           problem_type (int): Type of problem (e.g., regression or classification).
           batch_size (int): Number of samples per batch.
           learning_rate (float): Learning rate for optimization.
           max_iter (int): Maximum number of iterations (epochs).
           early_stopping (bool): Boolean flag for early stopping.
           verbose (bool): Boolean flag for verbosity during training.
           num_splits (int, optional): K-Fold Split of the data. Defaults to 10
           log (bool): Boolean flag for logging training progress.
       """
        lstm_net = LSTMNet(input_size, output_size, hidden_size, num_layers, dropout_prob)
        self.model = create_torch_model(lstm_net, problem_type, criterion_str, batch_size, learning_rate,
                                        max_iter, early_stopping, verbose, num_splits, log)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Trains the model on the provided dataset.

        Args:
            X (torch.Tensor): Input features.
            y (torch.Tensor): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generates predictions using the trained model.

        Args:
            X (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted output tensor.
        """
        return self.model.predict(X)

    def __str__(self) -> str:
        """
        Returns a string representation of the EasyLSTM instance.

        Returns:
            str: String representation including the number of layers, neurons per layer, and dropout probability.
        """
        return (f"EasyLSTM(Layers: {self.num_layers}, Neurons per Layer: {self.hidden_size}, "
                f"Dropout: {self.dropout_prob})")
