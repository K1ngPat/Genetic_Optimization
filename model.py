import torch
import torch.nn as nn
import numpy as np

# constants for connect-4
R = 6
C = 7
INPUT_DIM = (R, C)
HIDDEN_LAYERS = [25, 25]
OUTPUT_DIM = C

class Network(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_fn=nn.ReLU):
        super(Network, self).__init__()

        layers = [
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=0),
            activation_fn(),
            nn.Flatten(),
            nn.Linear((input_dim[0]-2) * (input_dim[1]-2), hidden_layers[0]),
            activation_fn()
        ]

        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation_fn())

        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sequential(x)
        return x
    
    # TODO: set value for column as -1 in output vector if column is filled



        
        
