import torch
import torch.nn as nn
import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention.ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.shape[0]

        # 连接线性层生成Q，K，V
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 通过split操作一次执行全部的Head
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # 求softmax(Q@K^T/sqrt(dk))@V
        scale = torch.sqrt((key.size(-1) // num_heads))
        context = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # 拼接各个Head输出为[Batch,SeqLen,512]
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 再来一个线性层，增加模型复杂度[Batch,SeqLen,512]==>[Batch,SeqLen,512]
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # ADD & Norm
        output = self.layer_norm(residual + output)

        return output
