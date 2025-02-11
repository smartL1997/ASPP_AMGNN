import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as GraphConvolution
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import TopKPooling

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

        # 定义池化层 ratio 默认 0.5
        self.pool1 = TopKPooling(nhid)
        self.pool2 = TopKPooling(nhid)

    def forward(self, x, adj, batch):

        # 第一层
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x, adj, _, batch, _, _ = self.pool1(x, adj, batch=batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 第二层
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x, adj, _, batch, _, _ = self.pool2(x, adj, batch=batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 第三层
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x3= torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class SFGCN(nn.Module):
    def __init__(self, nfeat,  nhid, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid, dropout)
        self.SGCN2 = GCN(nfeat, nhid, dropout)
        self.CGCN = GCN(nfeat, nhid, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(in_size=2*nhid, hidden_size=nhid)
        self.tanh = nn.Tanh()


    def forward(self, x, sadj, fadj, batch):
        emb1 = self.SGCN1(x, sadj, batch) # Special_GCN out1
        com1 = self.CGCN(x, sadj, batch)  # Common_GCN out1
        com2 = self.CGCN(x, fadj, batch)  # Common_GCN out2
        emb2 = self.SGCN2(x, fadj, batch) # Special_GCN out2
        Xcom = (com1 + com2) / 2

        # attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)

        output = emb
        return output, att, emb1, com1, com2, emb2, emb




