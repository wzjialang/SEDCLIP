"""
Graph-attention Temporal Adaptor (GTA) module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.nn.parameter import Parameter
from scipy.spatial.distance import pdist, squareform
import numpy as np


class DistanceAdj(nn.Module):
    '''
    A = exp(−(i − j)^2/λ)
    '''
    def __init__(self, lambda_gta):
        super(DistanceAdj, self).__init__()
        self.lambda_gta = lambda_gta

    def forward(self, batch_size, seq_len, device):
        arith = np.arange(seq_len).reshape(-1, 1)
        dist = pdist(arith, metric='cityblock').astype(np.float32)
        dist = torch.from_numpy(squareform(dist)).to(device)

        # Eq. (9)
        dist = torch.exp(-dist ** 2 / self.lambda_gta)
        dist = torch.unsqueeze(dist, 0).repeat(batch_size, 1, 1)  # [bs, seq_len, seq_len]

        return dist


class GraphAttentionLayer(nn.Module):
    """
    GAT
    """
    def __init__(self, in_features, out_features, dropout=0.1, bias=False, alpha=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.empty(in_features, out_features), requires_grad=True)

        self.alpha = alpha
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a, gain=np.sqrt(2.0))

        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input, adj):
        # To support batch operations
        Wh = input.matmul(self.weight)
        e = self.prepare_batch(Wh)  # (batch_size, number_nodes, number_nodes)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Eq. (10)
        attention = F.softmax(attention + torch.log(adj), dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Eq. (11)
        h_prime = torch.matmul(attention, Wh)
        output = F.elu(h_prime)

        return output

    def prepare_batch(self, Wh):
        B, N, E = Wh.shape

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, N, 1)

        e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GATMultiHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.1, nheads=1):
        super(GATMultiHeadLayer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = nheads
        self.attentions = [GraphAttentionLayer(in_dim, out_dim, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        # Eq. (12)
        out = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim * self.heads}, heads={self.heads})')


class GTA(nn.Module):
    """
    GTA
    """
    def __init__(self, cfg, d_model, n_heads, lambda_gta):
        super(GTA, self).__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.self_attn = GATMultiHeadLayer(
            d_model, d_model // self.n_heads,
            dropout=0.1, alpha=0.1, nheads=self.n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj(lambda_gta)

    def forward(self, x):
        # x: [B, T, C] -> adj over the T (time) axis
        adj = self.loc_adj(x.shape[0], x.shape[1], x.device)
        tmp = self.self_attn(x, adj)

        # Eq. (8)
        x = x + tmp
        x = self.norm(x)
        return x
