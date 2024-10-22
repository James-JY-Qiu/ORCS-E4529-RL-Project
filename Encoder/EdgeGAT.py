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
            activation,
    ):
        super(MultiLayerEdgeGAT, self).__init__()
        self.in_feats = in_feats
        self.edge_feats = edge_feats
        self.units = units
        self.num_heads = num_heads
        assert num_layers > 2, 'number of layers must be greater than 2'
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.edge_drop = edge_drop
        self.activation = activation

        assert units % num_heads == 0, 'units must be a multiple of num_heads'
        hidden_feats = units // num_heads

        self.node_features_embedding = nn.Linear(in_feats, units)
        self.edge_features_embedding = nn.Linear(edge_feats, units)

        edge_gat = [
            EdgeGATConv(in_feats=units, edge_feats=units, out_feats=hidden_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=activation),
        ] + [EdgeGATConv(in_feats=units, edge_feats=units, out_feats=hidden_feats, num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=activation) for _ in range(num_layers - 2)
        ] + [EdgeGATConv(in_feats=units, edge_feats=units, out_feats=units, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop, activation=None)
        ]
        self.edge_gat = nn.ModuleList(edge_gat)

        edge_embeddings = [nn.Linear(units, units) for _ in range(num_layers-1)]
        self.edge_embeddings = nn.ModuleList(edge_embeddings)
        self.edge_dropout = nn.Dropout(edge_drop)

    def forward(self, g):
        # 从图中提取出节点特征和边特征
        x = g.ndata['features']  # 获取 batched 图的节点特征
        w = g.edata['edge_features']  # 获取 batched 图的边特征
        edge_index = g.edges()

        # 初始化节点特征和边特征
        x = self.node_features_embedding(x)
        x = self.activation(x)
        w = self.edge_features_embedding(w)
        w = self.activation(w)

        for i in range(self.num_layers):
            # 每一层都传递图、节点特征和边特征
            x = self.edge_gat[i](g, x, w)
            # 需要将 (num_nodes, num_heads, hidden_dim) 展开为 (num_nodes, num_heads * hidden_dim)
            x = x.flatten(1)
            if i < self.num_layers - 1:
                w = self.edge_embeddings[i](w + x[edge_index[0]] + x[edge_index[1]])
                w = self.activation(w)
                w = self.edge_dropout(w)

        return x  # 返回最终输出，形状为 (num_nodes, num_heads * hidden_dim) 或 (num_nodes, out_feats)