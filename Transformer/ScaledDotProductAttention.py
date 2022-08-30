import torch
import torch.nn as nn
import padding_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, dk=None, attn_mask=None):
        """
        :param Q: Queries张量，形状为[B, L_q, D_q]
        :param K: Keys张量，形状为[B, L_k, D_k]
        :param V: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        :param dk: 缩放因子，一个浮点标量
        :param attn_mask: Masking张量，形状为[B, L_q, L_k]
        :return: 上下文张量和attetention张量
        """
        attention = torch.bmm(Q, K.transpose(1, 2))
        if dk is not None:
            attention = attention / dk
        if attn_mask is not None:
            attention = padding_mask.padding_mask(attention, attn_mask)
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, V)
        return context
