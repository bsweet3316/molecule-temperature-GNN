# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:13:25 2023

@author: Bryant's
"""


import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class Network(torch.nn.Module):
    
    # The following function will initialize the neural network model
    def __init__(self):
        
    
        super(Network, self).__init__()
    
        # The following will perform the 2 rounds of convolution (message passing)
        # for the GNN, the output is a 4 x 64 matrix
        self.conv1 = GCNConv(14, 128)
        self.conv2 = GCNConv(128, 64)
        
        # The following defines the Residual network fully connected layers.
        self.lin1 = torch.nn.Linear(256, 256)
        self.lin2 = torch.nn.Linear(256, 256)
        self.lin3 = torch.nn.Linear(256, 256)
        self.lin4 = torch.nn.Linear(256, 1)
        
    
    # The following will feed a single input through the network
    def forward(self, data):
        
        # Obtain the feature vectors and edge_index
        x, edge_index = data.x, data.edge_index
    
        # Perform the two rounds of convolution
        x = self.conv1(x, edge_index)
        x = self.conv2(x,edge_index)
        
        # Flatten the 4 x 64 matrix into a 256 vector
        x1 = torch.flatten(x)
        
        x2 = self.lin1(x1)
        x = self.lin2(x2)
        x = self.lin3(x)
        
        # The following represents the layer skipping for the residual network
        x = x + x1 + x2
        
        # Include a dropout layer before the final network layer.
        x = F.dropout(x, p=0.5, training = self.training)
        
        # Obtain the final output value.
        x = self.lin4(x)
        return x
        