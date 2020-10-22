import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self,
                input_dim,
                output_dim,
                layer_id,
                activation=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, h):

        h = self.linear(h)
        
        if self.activation:
            h = self.activation(h)
        return h