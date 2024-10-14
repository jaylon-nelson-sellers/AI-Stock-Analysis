from EasyTorch.EasyTorch import create_torch_model
import torch
import torch.nn as nn

# Check if CUDA is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    """
    A customizable neural network class.
    """

    def __init__(self, output_size: int, hidden_layer_sizes: tuple = (100,), dropout_prob: float = 0.0,
                 batch_norm: bool = True):
        """
        Initializes the network with the given parameters.

        :param output_size: Size of the output layer.
        :param hidden_layer_sizes: Sizes of the hidden layers.
        :param dropout_prob: Dropout probability.
        :param batch_norm: Whether to use batch normalization.
        """
        super(Net, self).__init__()
        layers = []
        self.batch_norm = batch_norm
        input_size = hidden_layer_sizes[0]

        # Create hidden layers
        for size in hidden_layer_sizes:
            layers.append(nn.LazyLinear(size))  # Add a linear layer with lazy initialization
            layers.append(nn.LeakyReLU())  # Add LeakyReLU activation function
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))  # Add batch normalization if enabled
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))  # Add dropout layer if dropout_prob > 0
            input_size = size

        # Add the final output layer
        layers.append(nn.LazyLinear(output_size))

        # Add a sigmoid activation if the output size is 1 (for binary classification)
        if output_size == 1:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)  # Combine layers into a sequential model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        :param x: Input tensor.
        :return: Output tensor.
        """
        if self.batch_norm and x.size(0) == 1:
            # Temporarily disable batch normalization for single batch
            self._disable_batch_norm()
            output = self.model(x)
            self._enable_batch_norm()
        else:
            output = self.model(x)
        return output

    def _disable_batch_norm(self) -> None:
        """
        Disable batch normalization by setting layers to evaluation mode.
        """
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()

    def _enable_batch_norm(self) -> None:
        """
        Enable batch normalization by setting layers back to training mode.
        """
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d):
                layer.train()


class EasyNeuralNet:
    """
    A class to simplify neural network creation and training using EasyTorch.
    """

    def __init__(self, output_size, hidden_layer_sizes: tuple = (100,), dropout_prob: float = 0.0,
                 batch_norm: bool = True, problem_type: int = 0, criterion_str="MSELoss", batch_size: int = 64,
                 learning_rate: float = 0.001, max_iter: int = 10000, early_stopping: bool = True,
                 verbose: bool = False, num_splits: int = 10, log: bool = False, image_bool=False):
        """
        Initializes the EasyNeuralNet with the given parameters.

        :param output_size: Size of the output layer.
        :param hidden_layer_sizes: Sizes of the hidden layers.
        :param dropout_prob: Dropout probability.
        :param batch_norm: Whether to use batch normalization.
        :param problem_type: Type of problem (e.g., regression, classification).
        :param criterion_str: Loss function.
        :param batch_size: Batch size for training.
        :param learning_rate: Learning rate for the optimizer.
        :param max_iter: Maximum number of iterations.
        :param early_stopping: Whether to use early stopping.
        :param verbose: Whether to print verbose logs.
        :param num_splits (float, optional): K-Fold Split of the data. Defaults to 10
        :param log: Whether to log training progress.
        :param image_bool: Flag to indicate if the input is image data.
        """
        net = Net(output_size, hidden_layer_sizes, dropout_prob, batch_norm)
        self.network = create_torch_model(net, problem_type, criterion_str, batch_size, learning_rate, max_iter,
                                          early_stopping, verbose, num_splits, log)
        self.image_bool = image_bool
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_prob = dropout_prob

    def fit(self,  X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fits the model to the given data.

        :param X: Input data.
        :param y: Target labels.
        :return: Trained model.
        """
        if self.image_bool:
            print("Image Type")
            y = y.view(-1, 1)  # Reshape target labels for image data
            print(y.shape)
        return self.network.fit(X, y)

    def predict(self,  X: torch.Tensor,) -> torch.Tensor:
        """
        Predicts the output for the given input data.

        :param X: Input data.
        :return: Predicted output.
        """
        return self.network.predict(X)

    def __str__(self)-> str:
        """
        String representation of the EasyNeuralNet.

        :return: String describing the network configuration.
        """
        return f'EasyNeuralNet Layers:{self.hidden_layer_sizes},Dropout:{self.dropout_prob})'
