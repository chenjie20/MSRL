import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from utils import *

class AttnLinearEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout):
        super(AttnLinearEncoder, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        # self.wn_linear = weight_norm(nn.Linear(input_dim, feature_dim))
        self.linear = nn.Linear(input_dim, feature_dim)
        self.wn_linear = weight_norm(module=self.linear)
        self.att_weights = nn.Parameter(torch.FloatTensor(feature_dim * 2, 1))
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.att_weights)

    #after
    def forward(self, x):
        z = self.wn_linear(x)
        e = torch.mm(z, self.att_weights[:self.feature_dim]) + torch.mm(z, self.att_weights[self.feature_dim:]).T
        e = F.relu(e)
        attention = F.softmax(e, dim=1)
        if self.dropout > 0:
            attention = self.dropout_layer(attention)

        z = torch.mm(attention, z) + z
        labels = F.softmax(z, dim=-1)

        return labels


