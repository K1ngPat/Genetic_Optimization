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
    def __init__(self, model_params=None, input_dim=INPUT_DIM, hidden_layers=HIDDEN_LAYERS, output_dim=C, activation_fn=nn.ReLU):
        super(Network, self).__init__()

        kernel_size = 2

        layers = [
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=0),
            activation_fn(),
            nn.Flatten(),
            nn.Linear((input_dim[0]-kernel_size+1) * (input_dim[1]-kernel_size+1), hidden_layers[0]),
            activation_fn()
        ]

        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation_fn())

        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.sequential = nn.Sequential(*layers)

        if(not model_params):
            self.load_state_dict(model_params)

    def forward(self, x):
        x = self.sequential(x)
        return x
    
    '''def get_parameters(self):
        p = [param for _, param in self.named_parameters()]
        params = [list(np.array(pa.tolist()).flatten()) for pa in p]
        params = params[2:-1]
        params = [item for sublist in params for item in sublist]
        return params'''
    
    def get_weights(self):
        return self.get_weights()
    
    # TODO: set value for column as -1 in output vector if column is filled



        
        
