import torch.nn as nn
import PositionalWiseFeedForward
import MultiHeadSelfAttention


class EncoderLayer(nn.Module):
    # Encoder的一层。
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention.MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward.PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # 进行多头自注意
        context = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        # context=[Batch,SeqLen,512]
        output = self.feed_forward(context)

        return output
