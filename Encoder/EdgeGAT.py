import torch.nn as nn
from dgl.nn import EdgeGATConv


class MultiLayerEdgeGAT(nn.Module):
    def __init__(
            self,
            in_feats,
            edge_feats,
            hidden_feats,
            out_feats,
            num_heads,
            num_layers,
            feat_drop,
            attn_drop,
            activation
    ):
        super(MultiLayerEdgeGAT, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        # 输入层
        self.layers.append(
            EdgeGATConv(
                in_feats=in_feats,
                edge_feats=edge_feats,
                out_feats=hidden_feats,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                activation=activation
            )
        )

        # 中间隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(
                EdgeGATConv(
                    in_feats=hidden_feats * num_heads,
                    edge_feats=edge_feats,
                    out_feats=hidden_feats,
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    activation=activation
                )
            )

        # 输出层
        self.layers.append(
            EdgeGATConv(
                in_feats=hidden_feats * num_heads,
                edge_feats=edge_feats,
                out_feats=out_feats,
                num_heads=1,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                activation=None
            )
        )

    def forward(self, g):
        # 从图中提取出节点特征和边特征
        h = g.ndata['features']  # 获取 batched 图的节点特征
        edge_weights = g.edata['edge_features']  # 获取 batched 图的边特征

        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weights)  # 每一层都传递图、节点特征和边特征
            # 需要将 (num_nodes, num_heads, hidden_dim) 展开为 (num_nodes, num_heads * hidden_dim)
            h = h.flatten(1)  # 展平多头的维度，适应下一层的输入维度

        return h  # 返回最终输出，形状为 (num_nodes, num_heads * hidden_dim) 或 (num_nodes, out_feats)