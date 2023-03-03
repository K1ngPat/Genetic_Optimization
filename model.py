import torch
import torch.nn as nn
import numpy as np
from io import BytesIO

# constants for connect-4
R = 6
C = 7
INPUT_DIM = (R, C)
HIDDEN_LAYERS = [25, 25]
OUTPUT_DIM = C

class Network(nn.Module):
    def __init__(self, weights1=None, weights2=None, mult1=1, mult2=1, input_dim=INPUT_DIM, hidden_layers=HIDDEN_LAYERS, output_dim=OUTPUT_DIM, activation_fn=nn.ReLU):
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

        # loading model weights if passed
        if weights1 != None and weights2 != None:
            state_dict1 = torch.load(BytesIO(weights1))
            state_dict2 = torch.load(BytesIO(weights2))
            new_state_dict = {}
            for k, v in state_dict1.items():
                new_key = "sequential." + k
                new_state_dict[new_key] = v*mult1 + state_dict2[k]*mult2
            self.load_state_dict(new_state_dict)

        elif weights1 != None:
            state_dict = torch.load(BytesIO(weights1))
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = "sequential." + k
                new_state_dict[new_key] = v*mult1
            self.load_state_dict(new_state_dict)

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
        buffer = BytesIO()
        torch.save(self.sequential.state_dict(), buffer)
        weights = buffer.getvalue()
        return weights

# create new Network from parent networks A and B with genetic ratio 1:9 using:
# C = Network(weights1=A.get_weights(), mult1=0.1, weights2=B.get_weights(), mult2=0.9)
# (A.get_weights() will be model_params in this case)

        
        
