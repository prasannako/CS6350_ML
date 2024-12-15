import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset


class Model(nn.Module):
    def __init__(self, in_features, out_features, depth, width, activation_function):
        super().__init__()
        self.activation_function = activation_function

        layers = [nn.Linear(in_features, width)] 
        layers += [nn.Linear(width, width) for _ in range(depth - 1)]  
        layers.append(nn.Linear(width, out_features)) 

        self.layers = nn.ModuleList(layers)

        for layer in self.layers:
            if activation_function == "tanh":
                nn.init.xavier_normal_(layer.weight)
            elif activation_function == "relu":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        for layer in self.layers[:-1]:
            if self.activation_function == "tanh":
                x = torch.tanh(layer(x))
            elif self.activation_function == "relu":
                x = F.relu(layer(x))
        return self.layers[-1](x)