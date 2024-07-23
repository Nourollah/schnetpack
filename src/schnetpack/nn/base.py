from typing import Callable, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_

from torch.nn.init import zeros_


__all__ = ["Dense"]


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y



class DynamicUNet(nn.Module):
    """
    A dynamic UNet-like module for atomistic systems.

    This module consists of an encoder (downscaling) part and a decoder (upscaling) part.
    The number of layers and the number of features in each layer can be controlled using parameters.

    Args:
        n_atom_basis (int): The number of features in the input.
        activation (Callable): The activation function to use.
        layers (int): The number of layers in the encoder and decoder.
    """

    def __init__(self, n_atom_basis, activation, layers):
        super(DynamicUNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Create the encoder layers
        for i in range(layers):
            # Each layer consists of two Dense layers
            # The number of features is doubled in each layer
            self.encoder.append(nn.Sequential(
                Dense(n_atom_basis, n_atom_basis * 2, activation=activation),
                Dense(n_atom_basis * 2, n_atom_basis * 2, activation=None),
            ))
            n_atom_basis *= 2

        # Create the decoder layers
        for i in range(layers):
            # Each layer consists of two Dense layers
            # The number of features is halved in each layer
            self.decoder.append(nn.Sequential(
                Dense(n_atom_basis, n_atom_basis // 2, activation=activation),
                Dense(n_atom_basis // 2, n_atom_basis // 2, activation=None),
            ))
            n_atom_basis //= 2

    def forward(self, x):
        """
        Forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        # Pass through the encoder
        for layer in self.encoder:
            x = layer(x)

        # Pass through the decoder
        for layer in self.decoder:
            x = layer(x)

        return x