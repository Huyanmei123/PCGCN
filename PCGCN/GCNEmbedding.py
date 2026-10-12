import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from config import *
import torch.nn.functional as F
import random
import numpy as np
from config import *

# class GCNemb(torch.nn.Module):
#     def __init__(self, num_node_features, output_dim, hidden_dim,cluster_num):
#         super(GCNemb, self).__init__()
#         self.centers = torch.nn.Parameter(torch.randn(cluster_num , output_dim))
#         self.conv1 = GCNConv(num_node_features, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.FC1 = torch.nn.Linear(hidden_dim, 64)
#         self.FC2 = torch.nn.Linear(64, output_dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, training=self.training, p=args.dropout)
#         x = self.conv2(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, training=self.training, p=args.dropout)
#         x = self.conv3(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, training=self.training, p=args.dropout)
#         x = self.FC1(x)
#         x = F.elu(x)
#         x = self.FC2(x)
#         return x
class GNN_layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GNN_layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size, self.output_size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.FloatTensor(output_size))
        nn.init.zeros_(self.bias)


    def forward(self, features, adj):
        # if active:
        #     support = self.act(F.linear(features, self.weight))
        # else:
        #     support = F.linear(features, self.weight)

        # output = torch.spmm(adj, support)
        # return output
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class GCNemb(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_dim,cluster_num):
        super(GCNemb, self).__init__()
        self.conv1 = GNN_layer(num_node_features, hidden_dim)
        self.conv2 = GNN_layer(hidden_dim, hidden_dim)
        self.conv3 = GNN_layer(hidden_dim, hidden_dim)
        
    
    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=args.dropout)
        x = self.conv2(x, adj)
        x = F.elu(x)
        if args.dataset=="ca-GrQc":
            x = F.normalize(x, dim=1, p=2)
        x = F.dropout(x, training=self.training, p=args.dropout)
        x = self.conv3(x, adj)
        if args.dataset== "netsci":
            x = F.softmax(x,dim=1)
        
         
        
        return x
         
        
   
