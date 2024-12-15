import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(TransformerBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        assert out_features % num_heads == 0, 'out_features must be a multiple of num_heads'
        self.head_dim = out_features // num_heads
        self.sqrt_head_dim = self.head_dim ** 0.5

        self.embedding = nn.Linear(in_features, out_features)
        self.query = nn.Linear(out_features, out_features)
        self.key = nn.Linear(out_features, out_features)
        self.value = nn.Linear(out_features, out_features)
        self.w_0 = nn.Linear(out_features, out_features)

        self.w_1 = nn.Linear(out_features, out_features)
        self.w_2 = nn.Linear(out_features, out_features)

    def embed(self, x):
        return self.embedding(x)

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

    def get_attention_score(self, q, k):
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.sqrt_head_dim, dim=-1)

    def get_attention_result(self, attention_score, v, x):
        attention_result = torch.matmul(attention_score, v)
        attention_result = attention_result.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.w_0(attention_result)

    def forward(self, x):
        # Embed the input
        input_embeds = self.embed(x)

        # Calculate query, key and value
        q = self.query(input_embeds)
        k = self.key(input_embeds)
        v = self.value(input_embeds)

        # Split the heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Calculate the attention score
        attention_score = self.get_attention_score(q, k)

        # Calculate the attention result
        attention_result = self.get_attention_result(attention_score, v, x)

        # MHA Residual connection
        mha_result = input_embeds + attention_result

        # Feed forward network
        ffn_result = self.w_2(torch.relu(self.w_1(mha_result)))

        # FNN Residual connection
        fnn_result = mha_result + ffn_result

        return fnn_result


class TransformerEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = [
            TransformerBlock(in_features, out_features, num_heads)
        ] + [
            TransformerBlock(out_features, out_features, num_heads)
            for _ in range(num_layers - 1)
        ]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
