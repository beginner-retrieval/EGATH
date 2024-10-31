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
        h_prime = torch.matmul(attention, h)

        return h_prime
# gen_A函数根据类别数num_classes、阈值t和邻接矩阵文件adj_file生成图的邻接矩阵
def gen_A(num_classes, t, adj_file):
    # 从文件中读取原始邻接矩阵和节点度数
    _adj = adj_file['adj']
    print(_adj.shape)
    _nums = adj_file['num']
    _nums = _nums[:, np.newaxis]
    # 将邻接矩阵归一化并应用阈值t，小于t的设置为0，否则设置为1
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    print(_adj.sum())
    # 将邻接矩阵标准化，并根据阈值t二值化
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    # 在邻接矩阵上加上单位矩阵，实现自连接
    _adj = _adj + np.identity(num_classes, int)
    return _adj
# 从邻接矩阵 A 生成规范化的邻接矩阵，这里使用了对角度矩阵 D 的平方根倒数进行规范化
def gen_adj(A):
    # # 接收一个邻接矩阵A
    A = A.squeeze()
    # 计算度矩阵的逆平方根
    D_sqrt_inv = torch.pow(A.sum(1).float(), -0.5)
    # 将度矩阵转换为对角矩阵形式
    D_sqrt_inv = torch.diag(D_sqrt_inv)
    # 对称归一化处理
    adj = torch.matmul(torch.matmul(D_sqrt_inv, A), D_sqrt_inv)
    return adj


class GATModel(BasicModule):
# 接受 flag（可能用于区分不同的配置或数据集）、隐藏层维度 hidden_dim、类别数 num_class 和邻接矩阵文件 adj_file 作为参数
    def __init__(self, flag, hidden_dim, num_class, adj_file):
        super(GATModel, self).__init__()
        # 定义了两个图卷积层，第一个从300维到1024维，第二个从1024维到隐藏层维度
        self.gat1 = GATLayer(300, 1024, dropout=opt.dropout, alpha=opt.alpha)
        self.gat2 = GATLayer(1024, hidden_dim, dropout=opt.dropout, alpha=opt.alpha)
        # 定义了一个LeakyReLU激活函数
        self.relu = nn.LeakyReLU(0.2)
        # 根据 flag 调整配置参数
        if flag == "nus":
            opt.deta = 0.01
        # 生成邻接矩阵
        _adj = gen_A(num_class, opt.deta, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
   # 定义前向传播函数
    def forward(self, x, inp):
        # 生成并使用归一化的邻接矩阵
        adj = gen_adj(self.A).detach()
        # 通过两个图卷积层和一个激活函数传播输入
        inp = self.gat1(inp, adj)
        inp = self.relu(inp)
        inp = self.gat2(inp, adj)
        # 转置输出，以匹配目标矩阵的形状
        inp = inp.transpose(0, 1)
        # 将转置后的输出与另一个输入x进行矩阵乘法，得到最终的类别预测
        x_class = torch.matmul(x, inp)
        return x_class
