import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


from layers import *

class TLC(nn.Module):
    def __init__(self, input_dims, feature_dim, dropout):
        super(TLC, self).__init__()
        self.dropout = dropout
        self.encoders = nn.ModuleList()
        for input_dim in input_dims:
            self.encoders.append(AttnLinearEncoder(input_dim, feature_dim, dropout))

    def forward(self, data_sets):
        labels_set = []

        for encoder, data in zip(self.encoders, data_sets):
            labels = encoder(data)
            labels_set.append(labels)
        labels = torch.mean(torch.stack(labels_set), dim=0)

        return labels, labels_set



