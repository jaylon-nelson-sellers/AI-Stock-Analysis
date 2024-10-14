import torch
import torch.nn as nn
import math
from EasyTorch.EasyTorch import compute_hidden_size, create_torch_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNNNet(nn.Module):
    def __init__(self, input_size, output_size, dimensions=1, num_channels=1, conv_layers_per_block=1, pool_blocks=1,
                 dense_layers=1, pool_method='max', batch_norm=True, conv_dropout=0.1, fully_con_dropout=0.5):
        """
        Initialize the EasyCNN model with specified architecture parameters.

        Args:
        input_size (int): The size of the input data.
        output_size (int): The size of the output layer.
        num_channels (int): Number of channels in the input data.
        conv_layers_per_block (int, optional): Number of convolutional layers per block. Defaults to 1.
        pool_blocks (int, optional): Number of pooling blocks in the model. Defaults to 1.
        dense_layers (int, optional): Number of dense layers after convolution blocks. Defaults to 1.
        pool_method (str, optional): Type of pooling layer ('max' or 'avg'). Defaults to 'max'.
        batch_norm (bool, optional): Whether to include batch normalization layers. Defaults to True.
        dropout (float, optional): Dropout rate for dropout layers. Defaults to 0.0.
        """
        super(CNNNet, self).__init__()
        assert conv_layers_per_block > 0 and pool_blocks > 0 and dense_layers > 0
        if dimensions == 1:
            self.conv_layers = self.create_1d_conv_layers(num_channels, conv_layers_per_block, pool_blocks, pool_method,
                                                          conv_dropout, batch_norm)
        elif dimensions == 2:
            self.conv_layers = self.create_2d_conv_layers(num_channels, conv_layers_per_block, pool_blocks, pool_method,
                                                          conv_dropout, batch_norm, )
        self.dense_layers = self.create_dense_layers(input_size, output_size, num_channels, pool_blocks, dense_layers,
                                                     fully_con_dropout, batch_norm)

    def create_1d_conv_layers(self, num_channels, conv_layers_per_block, pool_blocks, pool_method, conv_dropout,
                              batch_norm):
        """
        Create a sequential container of convolutional layers.

        Args:
        num_channels (int): Number of channels for convolutional layers.
        conv_layers_per_block (int): Number of convolution layers per block.
        pool_blocks (int): Number of pooling blocks.
        pool_method (str): Pooling method, 'avg' for average pooling and 'max' for max pooling.
        batch_norm (bool): Whether to include batch normalization layers.

        Returns:
        nn.Sequential: A sequential container of all convolutional layers including activation, pooling, and batch
        normalization layers.
        """
        layers = []
        for _ in range(pool_blocks):
            for _ in range(conv_layers_per_block):
                layers.append(nn.LazyConv1d(num_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.LeakyReLU())
                if conv_dropout > 0:
                    layers.append(nn.Dropout(conv_dropout))
                if batch_norm:
                    layers.append(nn.LazyBatchNorm1d())
            pool_layer = nn.AvgPool1d(kernel_size=2, stride=2) if pool_method == 'avg' else nn.MaxPool1d(kernel_size=2,
                                                                                                         stride=2)
            layers.append(pool_layer)
        return nn.Sequential(*layers)

    def create_2d_conv_layers(self, num_channels, conv_layers_per_block, pool_blocks, pool_method, conv_dropout,
                              batch_norm):
        """
        Create a sequential container of convolutional layers.

        Args:
        num_channels (int): Number of channels for convolutional layers.
        conv_layers_per_block (int): Number of convolution layers per block.
        pool_blocks (int): Number of pooling blocks.
        pool_method (str): Pooling method, 'avg' for average pooling and 'max' for max pooling.
        batch_norm (bool): Whether to include batch normalization layers.

        Returns:
        nn.Sequential: A sequential container of all convolutional layers including activation, pooling, and batch
        normalization layers.
        """
        layers = []
        for _ in range(pool_blocks):
            for _ in range(conv_layers_per_block):
                layers.append(nn.LazyConv2d(num_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.LeakyReLU())
                if conv_dropout > 0:
                    layers.append(nn.Dropout(conv_dropout))
                if batch_norm:
                    layers.append(nn.LazyBatchNorm2d())
            pool_layer = nn.AvgPool2d(kernel_size=2, stride=2) if pool_method == 'avg' else nn.MaxPool2d(kernel_size=2,
                                                                                                         stride=2)
            layers.append(pool_layer)
        return nn.Sequential(*layers)

    def create_dense_layers(self, input_size, output_size, num_channels, pool_blocks, dense_layers, fully_con_dropout,
                            batch_norm):
        """
        Create a sequential container of dense layers.

        Args:
        input_size (int): Initial size of input to dense layers.
        output_size (int): Size of the output layer.
        num_channels (int): Number of output channels from the last convolutional layer.
        pool_blocks (int): Number of pooling blocks, which affects the reduction in feature size.
        dense_layers (int): Number of dense layers.
        dropout (float): Dropout rate for dropout layers.
        batch_norm (bool): Whether to include batch normalization layers after each dense layer.

        Returns:
        nn.Sequential: A sequential container of all dense layers including dropout and batch normalization layers.
        """
        in_feats = math.floor(input_size / (2 ** pool_blocks)) * num_channels
        feats = compute_hidden_size(in_feats, output_size)
        layers = []
        for i in range(dense_layers):
            layers.append(nn.LazyLinear(feats))
            layers.append(nn.LeakyReLU())
            if fully_con_dropout > 0:
                layers.append(nn.Dropout(fully_con_dropout))
            if batch_norm:
                layers.append(nn.LazyBatchNorm1d())
        layers.append(nn.LazyLinear(output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
        x (torch.Tensor): Input tensor to process through the model.

        Returns:
        torch.Tensor: Output tensor after passing through convolutional and dense layers.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.dense_layers(x)


class EasyConvNet:
    def __init__(self, input_size, output_size, dimensions=1, num_channels=1, conv_layers_per_block=1, pool_blocks=1,
                 dense_layers=1, pool_method='max', batch_norm=True, dropout=0.0, problem_type: int = 0,
                 criterion_str="MSELoss",
                 batch_size: int = 64, learning_rate: float = 0.001, max_iter: int = 10000, early_stopping: bool = True,
                 verbose: bool = False, validation_split: int = 10, log: bool = False):
        net = CNNNet(input_size, output_size, dimensions, num_channels, conv_layers_per_block, pool_blocks,
                     dense_layers,
                     pool_method, batch_norm, dropout)
        self.network = create_torch_model(net, problem_type, criterion_str, batch_size, learning_rate, max_iter,
                                          early_stopping, verbose, validation_split, log)
        self.dimensions = dimensions
        self.num_channels = num_channels
        self.conv_layers_per_block = conv_layers_per_block
        self.pool_blocks = pool_blocks
        self.dense_layers = dense_layers

    def fit(self, X, y):
        print(X.shape)
        print(y.shape)
        return self.network.fit(X, y)

    def predict(self, X):
        return self.network.predict(X, )

    def __str__(self):
        return f"EasyConvNet(Dimensions:{self.dimensions}, Channels Per Layer: {self.num_channels}, " \
               f"Conv Layers Per Block:{self.conv_layers_per_block}, Pool Blocks:{self.pool_blocks}, " \
               f"Dense Layers:{self.dense_layers})"