import torch
import torch.nn as nn
from EasyTorch.EasyTorch import create_torch_model


class TransformerNet(nn.Module):
    """
    Transformer network for sequence processing.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_prob: float = 0.0):
        """
        Initializes the Transformer network.

        :param input_size: Size of the input features.
        :param output_size: Size of the output layer.
        :param hidden_size: Number of features in the hidden state.
        :param num_layers: Number of transformer layers.
        :param dropout_prob: Dropout probability.
        """
        super(TransformerNet, self).__init__()

        # Save initialization parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer to convert input features to hidden_size dimension
        self.embedding = nn.Linear(input_size, hidden_size)

        # Set number of heads for multi-head attention; use 1 head if hidden_size is less than 8
        self.heads = 1 if hidden_size < 8 else 8

        # Create a transformer encoder layer with specified hidden size, heads, and dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=self.heads, dropout=dropout_prob)

        # Stack the encoder layers to form the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final fully connected layer to convert hidden state to output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer network.

        :param x: Input tensor.
        :return: Output tensor.
        """
        # Convert input features to hidden size dimension
        x = self.embedding(x)

        # Create a mask to ignore padding positions during attention calculation
        src_key_padding_mask = torch.zeros((x.size(0), x.size(1)), dtype=torch.bool).to(x.device)

        # Permute dimensions to match transformer input requirements (sequence length, batch size, hidden size)
        x = x.permute(1, 0, 2)

        # Pass through the transformer encoder
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Take the last output of the sequence and pass it through the fully connected layer
        out = self.fc(out[-1, :, :])
        return out


class EasyTransformer:
    """
    Easy-to-use interface for training and using a Transformer network.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_prob: float = 0.0,
                 criterion_str: str = "MSELoss", problem_type: int = 0, batch_size: int = 64,
                 learning_rate: float = 0.001, max_iter: int = 10000, early_stopping: bool = True,
                 verbose: bool = False, num_splits: int = 10, log: bool = False):
        """
        Initializes the EasyTransformer with the given parameters.

        :param input_size: Size of the input features.
        :param output_size: Size of the output layer.
        :param hidden_size: Number of features in the hidden state.
        :param num_layers: Number of transformer layers.
        :param dropout_prob: Dropout probability.
        :param criterion_str: Loss function.
        :param problem_type: Type of problem (e.g., regression, classification).
        :param batch_size: Batch size for training.
        :param learning_rate: Learning rate for the optimizer.
        :param max_iter: Maximum number of iterations.
        :param early_stopping: Whether to use early stopping.
        :param verbose: Whether to print verbose logs.
        :param num_splits: Proportion of data to use for validation.
        :param log: Whether to log training progress.
        """
        # Initialize the TransformerNet
        net = TransformerNet(input_size, output_size, hidden_size, num_layers, dropout_prob)

        # Create an EasyTorch model with the given network and training parameters
        self.network = create_torch_model(net, problem_type, criterion_str, batch_size, learning_rate, max_iter,
                                          early_stopping, verbose, num_splits, log)

        # Save initialization parameters
        self.log = log
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.heads = 1 if hidden_size < 8 else 8

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fits the model to the given data.

        :param X: Input data.
        :param y: Target labels.
        :return: Trained model.
        """
        if self.log:
            print("Training started...")

        # Train the network with the provided data
        result = self.network.fit(X, y)

        if self.log:
            print("Training completed.")

        return result

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output for the given input data.

        :param X: Input data.
        :return: Predicted output.
        """
        if self.log:
            print("Prediction started...")

        # Predict the output using the trained network
        result = self.network.predict(X)

        if self.log:
            print("Prediction completed.")

        return result

    def __str__(self) -> str:
        """

        :return: String describing the network configuration.
        """
        return (f"EasyTransformer(Layers: {self.num_layers}, Heads: {self.heads}, "
                f"Neurons per Layer: {self.hidden_size}, Dropout: {self.dropout_prob})")
