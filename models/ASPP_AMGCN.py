import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn.gnn_models import Model as GCN_Model
from gnn.AM_GCN_pro import SFGCN
from models.CBAM_1D import CBAMBlock
from torch_cluster import knn_graph


class attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, enc_output):

        energy = torch.tanh(self.attn(enc_output))

        attention = self.v(energy)

        scores = F.softmax(attention, dim=1)
        out = enc_output * scores

        return torch.sum(out, dim=1), scores


class AsppBlock(nn.Module):
    def __init__(self, in_channel=4, out_channel=4):
        super(AsppBlock, self).__init__()

        self.atrous_block1 = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=1,
            dilation=1)

        self.atrous_block6 = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=6,
            dilation=6)

        self.atrous_block12 = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=12,
            dilation=12)

        self.atrous_block18 = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            padding=18,
            dilation=18)

        self.bn1 = nn.BatchNorm1d(num_features=out_channel)
        self.bn2 = nn.BatchNorm1d(num_features=out_channel)
        self.bn3 = nn.BatchNorm1d(num_features=out_channel)
        self.bn4 = nn.BatchNorm1d(num_features=out_channel)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 四个堆叠的空洞卷积
        atrous_block1 = self.relu(self.bn1(self.atrous_block1(x)))
        atrous_block6 = self.relu(self.bn2(self.atrous_block6(x)))
        atrous_block12 = self.relu(self.bn3(self.atrous_block12(x)))
        atrous_block18 = self.relu(self.bn4(self.atrous_block18(x)))
        x = torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        return x


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # ASPP 块
        self.aspp_block = AsppBlock(in_channels, out_channels)

        self.conv_block1 = nn.Conv1d(
            in_channels=out_channels*4,
            out_channels=out_channels*4,
            kernel_size=5,
            padding=2,
            dilation=1)

        self.conv_block2 = nn.Conv1d(
            in_channels=out_channels*4,
            out_channels=out_channels*4,
            kernel_size=3,
            padding=1,
            dilation=1)

        self.bn1 = nn.BatchNorm1d(num_features=out_channels*4)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels*4)

        self.cbma = CBAMBlock(in_channels=out_channels*4)

        self.relu = nn.ReLU(inplace=False)
        self.pooling = nn.MaxPool1d(4)

    def forward(self, x):

        x = self.aspp_block(x)
        x = self.pooling(x)

        identity = x

        out = self.relu(self.bn1(self.conv_block1(x)))
        out = self.relu(self.bn2(self.conv_block2(out)))

        out = self.cbma(out)

        out = out + identity
        return out

# 定义深度残差网络
class Model(nn.Module):
    def __init__(self,  args):
        super(Model, self).__init__()
        self.conv_start = nn.Conv1d(1, 4, kernel_size=3, padding=1, dilation=1)
        self.bn_start = nn.BatchNorm1d(num_features=4)

        # 残差块
        self.block = nn.Sequential(ResidualBlock(4, 4),
                                   ResidualBlock(16, 16),
                                   ResidualBlock(64, 64))
        self.relu = nn.ReLU(inplace=False)
        self.gmp_lead = nn.AdaptiveMaxPool1d(1)
        self.gap_lead = nn.AdaptiveAvgPool1d(1)
        self.leads_attn = attention(enc_hid_dim=512, dec_hid_dim=64)
        self.conv2D_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 11))
        self.bn_conv2D_1 = nn.BatchNorm2d(num_features=8)
        self.conv2D_2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(3, 5))
        self.bn_conv2D_2 = nn.BatchNorm2d(num_features=64)
        self.pooling_2D = nn.MaxPool2d((2, 2))
        self.gmp_all = nn.AdaptiveMaxPool2d((1, 1))
        self.gap_all = nn.AdaptiveAvgPool2d((1, 1))
        self.gcn = GCN_Model(args)
        self.sfgcn = SFGCN(nfeat=512, nhid=256, dropout=0.0)
        self.fc = nn.Linear(in_features=512, out_features=5)

    def forward(self, data):

        x, edge_index, batch = data.x.double(),\
                               data.edge_index, \
                               data.batch

        x = x.reshape(-1, 1000).unsqueeze(1)
        x = self.relu(self.bn_start(self.conv_start(x)))
        x = self.block(x)
        gmp_x = self.gmp_lead(x)
        gap_x = self.gap_lead(x)
        x = torch.cat([gmp_x, gap_x], dim=1)
        x = x.squeeze(2).reshape(-1, 12, 512)

        # AM_GCN
        x = x.reshape(-1, 512, 1).squeeze(2)
        # 拓扑图
        sadj = edge_index
        # 特征图
        fadj = knn_graph(x, k=3, loop=False)
        output, att, emb1, com1, com2, emb2, emb = self.sfgcn(x, sadj, fadj, batch)
        x = output

        x = self.fc(x)
        return x, att, emb1, com1, com2, emb2, emb

if __name__ == "__main__":
    net = Model()
    sample_x = torch.randn(32, 12, 1000)
    x = net(sample_x)



