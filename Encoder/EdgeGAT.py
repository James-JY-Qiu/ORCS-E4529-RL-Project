import torch.nn as nn
from dgl.nn import EdgeGATConv


class MultiLayerEdgeGAT(nn.Module):
    def __init__(
            self,
            in_feats,
            edge_feats,
            units,
            num_heads,
            num_layers,
            feat_drop,
            attn_drop,
            edge_drop,
            activation
    ):
        super(MultiLayerEdgeGAT, self).__init__()
        self.in_feats = in_feats
        self.edge_feats = edge_feats
        self.units = units
        self.num_heads = num_heads
        assert num_layers >= 2, 'number of layers must be at least 2'
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.edge_drop = edge_drop
        self.activation = activation

        assert units % num_heads == 0, 'units must be a multiple of num_heads'
        hidden_feats = units // num_heads

        edge_gat = [
            EdgeGATConv(in_feats=in_feats, edge_feats=edge_feats, out_feats=hidden_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=activation),
        ] + [EdgeGATConv(in_feats=units, edge_feats=edge_feats, out_feats=hidden_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=activation) for _ in range(num_layers - 2)
        ] + [EdgeGATConv(in_feats=units, edge_feats=edge_feats, out_feats=units, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop, activation=activation)
        ]
        self.edge_gat = nn.ModuleList(edge_gat)

    def forward(self, g):
        # 从图中提取出节点特征和边特征
        x = g.ndata['features']  # 获取 batched 图的节点特征
        w = g.edata['edge_features']  # 获取 batched 图的边特征

        for i in range(self.num_layers):
            # 每一层都传递图、节点特征和边特征
            x = self.edge_gat[i](g, x, w).flatten(1)

        return x  # 返回最终输出，形状为 (num_nodes, num_heads * hidden_dim) 或 (num_nodes, out_feats)