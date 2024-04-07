from math import inf
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hgcn.layers.hyp_layers import HypLinear
from hgcn.manifolds import PoincareBall




class HyperbolicGraphConstructor2(nn.Module):
    def __init__(self, nnodes, k, dim, device, nclass, args):
        super(HyperbolicGraphConstructor2, self).__init__()
        self.num_features = nnodes
        self.nhid = dim
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.dropout = args.dropout
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0
        self.alpha = 3
        self.device = device

        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin = nn.Linear(nnodes, nnodes)
        self.hyp_lin = HypLinear(self.manifold, 10882, nnodes, self.c, 0.01, True)
        self.hyp_lin1 = HypLinear(self.manifold, dim, dim, self.c, 0.01, True)
        self.hyp_lin2 = HypLinear(self.manifold, dim, dim, self.c, 0.01, True)

    def forward(self, idx, dist_metrix, x, avg_metrix):
        x = x.to(self.device)
        dist_metrix = dist_metrix.to(self.device)
        avg_metrix = avg_metrix.to(self.device)
        hyp_x = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), self.c), self.c)
        # h_x = hyp_x.transpose(0, 1)
        h_x = self.hyp_lin(hyp_x)
        hyp_avg = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(avg_metrix, self.c), self.c), self.c)
        for i in range(len(hyp_x)):
            for j in range(len(hyp_x)):
                dist_metrix[i] = self.manifold.sqdist(hyp_x[i].view(1, -1), hyp_avg[i].view(1, -1), self.c).clone().detach()
        dist_metrix = torch.nn.functional.normalize(-1 * dist_metrix)
        dist_metrix = self.lin(dist_metrix)
        a = torch.softmax(dist_metrix, dim=0)
        hyp_x = self.manifold.mobius_matvec(hyp_x.transpose(0, 1), a, self.c)
        x = self.manifold.proj_tan0(self.manifold.logmap0(hyp_x, c=self.c), c=self.c)
        return x

    def loss(self, pred, label, param_metrix):
        loss = 0
        for i in range(len(pred)):
            mu = 1
            loss += mu * torch.norm(pred[i] - label[i]) ** 2 \
                # + torch.norm(pred[i] - param_metrix[i]) ** 2
        return loss \
               # / len(pred)


def matrix2list(matrix):
    result = []
    N = len(matrix)
    for i in range(N):
        for j in range(N):
            if matrix[i][j] and matrix[i][j] != inf:
                result.append((i, j))
    result = torch.tensor(result).transpose(-1, -2)
    return result


def normalize_graph_adj(mx, device):
    """Row-normalize sparse matrix"""
    mx = mx.cpu()
    rowsum = np.array(mx.detach().sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx.detach())
    mx = torch.tensor(mx)
    return mx.to(device)
