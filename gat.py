import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from config import opt
from .basic_module import BasicModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h
        return h_prime

def gen_A(num_classes, t, adj_file):
    _adj = adj_file['adj']
    print(_adj.shape)
    _nums = adj_file['num']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    print(_adj.sum())
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, int)
    return _adj
def gen_adj(A):
    A = A.squeeze()
    D_sqrt_inv = torch.pow(A.sum(1).float(), -0.5)
    D_sqrt_inv = torch.diag(D_sqrt_inv)
    adj = torch.matmul(torch.matmul(D_sqrt_inv, A), D_sqrt_inv)
    return adj
class GATModel(BasicModule):
    def __init__(self, flag, hidden_dim, num_class, adj_file):
        super(GATModel, self).__init__()
        self.gat1 = GATLayer(300, 1024, dropout=opt.dropout, alpha=opt.alpha)
        self.gat2 = GATLayer(1024, hidden_dim, dropout=opt.dropout, alpha=opt.alpha)
        self.relu = nn.LeakyReLU(0.2)
        if flag == "nus":
            opt.deta = 0.01
        _adj = gen_A(num_class, opt.deta, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
    def forward(self, x, inp):
        adj = gen_adj(self.A).detach()
        inp = self.gat1(inp, adj)
        inp = self.relu(inp)
        inp = self.gat2(inp, adj)
        inp = inp.transpose(0, 1)
        x_class = torch.matmul(x, inp)
        return x_class
