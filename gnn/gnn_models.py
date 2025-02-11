import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from gnn.layers import GCN, HGPSLPool


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = 512
        self.nhid = 256
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        # 图卷积
        self.conv1 = GCN(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        self.edge_weight = torch.nn.Parameter(torch.FloatTensor(26, 1),
                                              requires_grad=True)
        self.edge_weight.data.fill_(1)

    def forward(self, data_x, data_edge_index, data_batch):
        x, edge_index, batch = data_x, data_edge_index, data_batch

        edge_weight = self.edge_weight
        _edge_weight = edge_weight
        for i in range(edge_index.shape[-1] // edge_weight.shape[0] - 1):
            edge_weight = torch.cat((edge_weight, _edge_weight), dim=0)
        edge_attr = edge_weight

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x,_ = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = F.dropout(x3, p=0.5, training=self.training)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        return x
